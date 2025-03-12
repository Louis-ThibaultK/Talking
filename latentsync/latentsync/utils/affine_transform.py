# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch


def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0)
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias


class AlignRestore(object):
    def __init__(self, align_points=3):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None

    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        cv2.imwrite("restored.jpg", restored_img)
        cv2.imwrite("aligned.jpg", aligned_face)
        return aligned_face, restored_img

    def align_warp_face(self, img, lmks3, smooth=True, border_mode="constant"):
        affine_matrix, self.p_bias = transformation_from_points(lmks3, self.face_template, smooth, self.p_bias)
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT

        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=border_mode,
            borderValue=[127, 127, 127],
        )
        return cropped_face, affine_matrix

    def align_warp_face2(self, img, landmark, border_mode="constant"):
        affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT
        cropped_face = cv2.warpAffine(
            img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132)
        )
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        if self.upscale_factor > 1:
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4)
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8)
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        if np.max(upsample_img) > 256:
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        return upsample_img
    
    def restore_img_gpu(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        #  1. 使用 CUDA 加速插值
        input_img_gpu = cv2.cuda_GpuMat()
        input_img_gpu.upload(input_img)
        upsample_img_gpu = cv2.cuda.resize(input_img_gpu, (w_up, h_up), interpolation=cv2.INTER_LINEAR)

        #  2. 计算仿射变换
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        if self.upscale_factor > 1:
            inverse_affine[:, 2] += 0.5 * self.upscale_factor

        #  3. 使用 CUDA 加速 warpAffine
        face_gpu = cv2.cuda_GpuMat()
        face_gpu.upload(face)
        inv_restored_gpu = cv2.cuda.warpAffine(face_gpu, inverse_affine, (w_up, h_up), flags=cv2.INTER_LINEAR)

        #  4. 生成 mask 并用 CUDA 处理
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        mask_gpu = cv2.cuda_GpuMat()
        mask_gpu.upload(mask)
        inv_mask_gpu = cv2.cuda.warpAffine(mask_gpu, inverse_affine, (w_up, h_up))

        #  5. CUDA 加速形态学腐蚀
        kernel_size = max(1, int(2 * self.upscale_factor))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        kernel_gpu = cv2.cuda_GpuMat()
        kernel_gpu.upload(kernel)
        inv_mask_erosion_gpu = cv2.cuda.erode(inv_mask_gpu, kernel_gpu)

        #  6. 计算 face blending
        inv_mask_erosion = inv_mask_erosion_gpu.download()
        pasted_face_gpu = cv2.cuda.multiply(inv_mask_erosion_gpu, inv_restored_gpu)

        #  7. 计算模糊 mask（CUDA）
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = max(1, w_edge * 2)
        kernel_blur = np.ones((erosion_radius, erosion_radius), np.uint8)
        kernel_blur_gpu = cv2.cuda_GpuMat()
        kernel_blur_gpu.upload(kernel_blur)
        inv_mask_center_gpu = cv2.cuda.erode(inv_mask_erosion_gpu, kernel_blur_gpu)

        blur_size = max(1, w_edge * 2)
        inv_soft_mask_gpu = cv2.cuda.GaussianBlur(inv_mask_center_gpu, (blur_size + 1, blur_size + 1), 0)

        #  8. 计算融合
        inv_soft_mask = inv_soft_mask_gpu.download()
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = upsample_img_gpu.download()

        upsample_img = inv_soft_mask * pasted_face_gpu.download() + (1 - inv_soft_mask) * upsample_img

        #  9. 避免不必要的 dtype 转换
        upsample_img = np.clip(upsample_img, 0, 255).astype(np.uint8)

        return upsample_img


class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update
