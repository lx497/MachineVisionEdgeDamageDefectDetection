"""
design by lx497
2022.12.25
"""
import math
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

name = ""
end_image_root = "result\\"

def add_xy(add_lines):
    """
    增加点数，进行拟合，让长度长的直线，权重更大
    :param add_lines:
    :return:
    """
    for line in add_lines:
        x1, y1, x2, y2 = line
        r = pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)
        if r < 50:
            continue
        xy_list = []
        if x2 - x1 == 0:
            print("直线是竖直的")
            if y2 > y1:
                max_y = y2
                min_y = y1
            else:
                min_y = y2
                max_y = y1
            while min_y < max_y:
                min_y = min_y + 40
                xy_list.append([x1, min_y])

        elif y2 - y1 == 0:
            print("直线是水平的")
            if x2 > x1:
                max_x = x2
                min_x = x1
            else:
                min_x = x2
                max_x = x1
            while min_x < max_x:
                min_x = min_x + 40
                xy_list.append([min_x, y1])

        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            # 求反正切，再将得到的弧度转换为度
            result_a = np.arctan(k) * 57.29577
            if -45 < result_a < 45:
                if x2 > x1:
                    max_x = x2
                    min_x = x1
                else:
                    min_x = x2
                    max_x = x1
                while min_x < max_x:
                    min_x = min_x + 40
                    xy_list.append([min_x, int(k * min_x + b)])
            else:
                if y2 > y1:
                    max_y = y2
                    min_y = y1
                else:
                    min_y = y2
                    max_y = y1
                while min_y < max_y:
                    min_y = min_y + 40
                    xy_list.append([int((y1 - b) / k), min_y])
        return xy_list


def line_detect(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    line_vertical = []
    line_horizontal = []

    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((10, 10), np.uint8)
    # 3图像的开闭运算
    cv0pen = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)  # 开运算

    edges = cv2.Canny(cv0pen, 50, 150, apertureSize=3)

    # 显示图片
    cv2.imwrite(end_image_root+"edges\\"+name, edges)
    # 检测白线  这里是设置检测直线的条件，可以去读一读HoughLinesP()函数，然后根据自己的要求设置检测条件
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=10, maxLineGap=10)
    i = 1
    # 对通过霍夫变换得到的数据进行遍历
    for line in lines:
        # newlines1 = lines[:, 0, :]
        # print("line[" + str(i - 1) + "]=", line)
        x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
        # 转换为浮点数，计算斜率
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        # print("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
        if x2 - x1 == 0:
            # print("直线是竖直的")

            line_vertical.append(line[0])

        elif y2 - y1 == 0:
            # print("直线是水平的")

            line_horizontal.append(line[0])

        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.29577
            if -45 < result < 45:
                line_horizontal.append(line[0])
            else:
                line_vertical.append(line[0])
            # print("直线倾斜角度为：" + str(result) + "度")
        i = i + 1
    #   显示最后的成果图
    cv2.imwrite(end_image_root+"line_detect\\"+name, image)
    return line_horizontal, line_vertical


def cal_point(point_list):
    """

    :param point_list:
    :return:
    """
    xy = []
    for point in point_list:
        xy.append([point[0], point[1]])
        xy.append([point[2], point[3]])
    return xy


def cal_xy_point(xy_points):
    """

    :param xy_points:
    :return:
    """
    point_x = []
    point_y = []
    for point in xy_points:
        point_x.append(point[0])
        point_y.append(point[1])
    return point_x, point_y


def linefit(x, y):
    """
    拟合曲线
    :param x:
    :param y:
    :return:
    """
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(0, int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sxy += x[i] * y[i]
    a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    b = (sy - a * sx) / N
    r = abs(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
    return a, b, r


# 定义直线拟合函数
def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)

    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])

    return np.linalg.solve(A, b)


def cal_a_and_b(points, flag):
    """
    计算最大最小值
    :param points:
    :param flag: falg为1 为垂直检测，0为水平检测
    :return:
    """
    print(points)
    max_value = points[0][flag]
    min_value = points[0][flag]
    min_point = points[0]
    max_point = points[0]
    for point in points:
        print(point[flag])
        if point[flag] > max_value:
            max_value = point[flag]
            max_point = point
        if point[flag] < min_value:
            min_value = point[flag]
            min_point = point
    x1, y1 = max_point
    x2, y2 = min_point
    k = -(y2 - y1) / (x2 - x1)
    c = y2 - k * x2
    return k, c


def del_point(lines):
    temp_list = []
    for line in lines:
        x1, y1, x2, y2 = line
        r = pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)
        if r < 100:
            continue
        else:
            temp_list.append(line)
    return temp_list


def cal_k_ave(lines):
    """
    求斜率平均值
    :param lines:
    :return:
    """
    new_lines = []
    total_length = 0  # 全部总长
    bia_total_len = 0  # 斜线总长
    bia_lines = []  # 放斜线坐标
    bia_line = []
    temp_lines = []
    end_k = 0.0
    end_b = 0.0
    angle = 0.0
    p = 0.0
    b = 0.0
    p_v = 0.0
    p_bia = 0.0
    for line in lines:
        x1, y1, x2, y2 = line
        r = int(pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5))
        new_line = list(line)
        total_length = total_length + r
        new_line.append(r)
        new_lines.append(new_line)

    for new_line in new_lines:
        x1, y1, x2, y2, r = new_line
        p = r / total_length
        if x1 - x2 != 0:
            p_bia = p_bia + p  # 概率加
            bia_total_len = bia_total_len + r
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            bia_line = [*new_line, k, b]

            bia_lines.append(bia_line)
        else:
            p_v = p_v + p  # 概率加
    if p_v > 0.5:
        return []
    else:
        rr = 0
        for bia_line in bia_lines:
            x1, y1, x2, y2, r, k, b = bia_line
            p = r / bia_total_len
            end_k = end_k + k * p
            end_b = end_b + b * p
            rr = rr + p
            if abs(np.arctan(k) * 57.29577 - np.arctan(end_k) * 57.29577) > 20:
                bia_lines.remove(bia_line)
                for temp in bia_lines:
                    temp_lines.append(temp[0:4])

                end_k, end_b = cal_k_ave(temp_lines)
        return [end_k, end_b]


def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c


def get_line_cross_point(line1, line2):
    """
    计算交点
    :param line1:
    :param line2:
    :return:
    """
    a0, b0, c0 = calc_abc_from_line_2d(*line1)
    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return x, y


def cal_k_ave_h(lines):
    """
    求斜率平均值
    :param lines:
    :return:
    """
    new_lines = []
    total_length = 0  # 全部总长
    end_k = 0.0
    end_b = 0.0
    end_lines = []
    temp_lines = []

    for line in lines:
        x1, y1, x2, y2 = line
        r = int(pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5))
        new_line = list(line)
        total_length = total_length + r
        new_line.append(r)
        new_lines.append(new_line)
    for new_line in new_lines:
        x1, y1, x2, y2, r = new_line
        p = r / total_length

        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        temp_line = [*new_line, k, b]
        end_lines.append(temp_line)

        end_k = end_k + k * p
        end_b = end_b + b * p
    for end_line in end_lines:
        x1, y1, x2, y2, r, k, b = end_line
        if abs(np.arctan(k) * 57.29577 - np.arctan(end_k) * 57.29577) > 10:
            end_lines.remove(end_line)
            for temp in end_lines:
                temp_lines.append(temp[0:4])

            end_k, end_b = cal_k_ave_h(temp_lines)

    return [end_k, end_b]


def main(path, file_name):
    # 画点参数
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 2

    temp_x = []
    temp_y = []
    line_l = []  # 左
    line_r = []  # 右
    line_t = []  # top
    line_d = []  # down
    # 读入图片
    src = cv2.imread(path)
    image_temp = src.copy()
    h, w, _ = image_temp.shape

    # 显示原始图片
    cv2.imwrite(end_image_root+"input image\\" + file_name, src)

    # 图像二值化
    image_bin_temp = src.copy()
    img = cv2.cvtColor(image_bin_temp, cv2.COLOR_BGR2GRAY)

    ret2, image_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 调用函数
    horizontal, vertical = line_detect(src)

    # 遍历horizontal求y的均值进行判断是上端还是下端
    for temp in horizontal:
        temp_x.append(temp[1])
        temp_x.append(temp[3])
    average_h = np.mean(temp_x)
    for temp in horizontal:
        if temp[1] > average_h and temp[3] > average_h:
            line_d.append(temp)
        else:
            line_t.append(temp)

    for temp in vertical:
        temp_y.append(temp[0])
        temp_y.append(temp[2])
    average_v = np.mean(temp_y)
    for temp in vertical:
        if temp[0] > average_v and temp[2] > average_v:
            line_r.append(temp)
        else:
            line_l.append(temp)

    # 转化为xy型式，便于拟合
    point_t = cal_point(line_t)
    point_d = cal_point(line_d)
    point_l = cal_point(line_l)
    point_r = cal_point(line_r)

    # 转化形式，用于拟合
    point_t_x, point_t_y = cal_xy_point(point_t)
    point_d_x, point_d_y = cal_xy_point(point_d)
    point_l_x, point_l_y = cal_xy_point(point_l)
    point_r_x, point_r_y = cal_xy_point(point_r)
    #
    total_point = []
    re_l = cal_k_ave(line_l)
    if len(re_l):
        al, bl = re_l
        point_l1 = (int(-bl / al) - 2, 0)  # 用来画直线
        point_l2 = (int((h - bl) / al) - 2, h)
        cv2.line(image_temp, point_l1, point_l2, (0, 255, 0), thickness=2)
    else:

        point_l1 = (point_l_x[0] - 2, 0)  # 用来画直线
        point_l2 = (point_l_x[0] - 2, h)

        cv2.line(image_temp, point_l1, point_l2, (0, 255, 0), thickness=2)

    re_r = cal_k_ave(line_r)

    if len(re_r):
        ar, br = re_r
        point_r1 = (int(-br / ar), 0)  # 用来画直线
        point_r2 = (int((h - br) / ar), h)
        cv2.line(image_temp, point_r1, point_r2, (0, 255, 0), thickness=2)


    else:
        point_r1 = (point_r_x[0] + 2, 0)  # 用来画直线
        point_r2 = (point_r_x[0] + 2, h)
        cv2.line(image_temp, point_r1, point_r2, (0, 255, 0), thickness=2)

    re_t = cal_k_ave_h(line_t)
    if len(re_t):
        tr_point = []
        tl_point = []
        at, bt = re_t
        point_t1 = (0, int(bt) - 2)  # 用来画直线
        point_t2 = (w, int(w * at + bt) - 2)
        cv2.line(image_temp, point_t1, point_t2, (0, 255, 0), thickness=2)
        # 垂直时计算交点
        if point_r1[0] - point_r2[0] == 0:
            temp_xx = point_r1[0]
            temp_yy = int(round(point_r1[0] * at + bt, 0))
            tr_point.append(temp_xx)
            tr_point.append(temp_yy)
        else:
            # 算出交点
            tr_point = get_line_cross_point([*list(point_t1), *list(point_t2)], [*list(point_r1), *list(point_r2)])
        # 取整四舍五入
        point = (int(round(tr_point[0], 0)), int(round(tr_point[1], 0)))  # 点的坐标。画点实际上就是画半径很小的实心圆。
        cv2.circle(image_temp, point, point_size, point_color, thickness)
        total_point.append(list(point))

        # 垂直时计算交点
        if point_l1[0] - point_l2[0] == 0:
            temp_xx = point_l1[0]
            temp_yy = int(round(point_l1[0] * at + bt, 0))
            tl_point.append(temp_xx)
            tl_point.append(temp_yy)
        else:
            tl_point = get_line_cross_point([*list(point_t1), *list(point_t2)], [*list(point_l1), *list(point_l2)])
        point = (int(round(tl_point[0], 0)), int(round(tl_point[1], 0)))  # 点的坐标。画点实际上就是画半径很小的实心圆。
        cv2.circle(image_temp, point, point_size, point_color, thickness)
        total_point.append(list(point))


    else:
        print("t error")

    re_d = cal_k_ave_h(line_d)
    if len(re_d):
        dr_point = []
        dl_point = []
        ad, bd = re_d
        point_d1 = (0, int(bd) + 2)  # 用来画直线
        point_d2 = (w, int(w * ad + bd) + 2)
        cv2.line(image_temp, point_d1, point_d2, (0, 255, 0), thickness=2)

        # 垂直时计算交点
        if point_r1[0] - point_r2[0] == 0:
            temp_xx = point_r1[0]
            temp_yy = int(round(point_r1[0] * ad + bd, 0))
            dr_point.append(temp_xx)
            dr_point.append(temp_yy)
        else:
            # 算出交点
            dr_point = get_line_cross_point([*list(point_d1), *list(point_d2)], [*list(point_r1), *list(point_r2)])
        # 取整四舍五入
        point = (int(round(dr_point[0], 0)), int(round(dr_point[1], 0)))  # 点的坐标。画点实际上就是画半径很小的实心圆。
        cv2.circle(image_temp, point, point_size, point_color, thickness)
        total_point.append(list(point))

        # 垂直时计算交点
        if point_l1[0] - point_l2[0] == 0:
            temp_xx = point_l1[0]
            temp_yy = int(round(point_l1[0] * ad + bd, 0))
            dl_point.append(temp_xx)
            dl_point.append(temp_yy)
        else:
            dl_point = get_line_cross_point([*list(point_d1), *list(point_d2)], [*list(point_l1), *list(point_l2)])
        point = (int(round(dl_point[0], 0)), int(round(dl_point[1], 0)))  # 点的坐标。画点实际上就是画半径很小的实心圆。
        cv2.circle(image_temp, point, point_size, point_color, thickness)
        total_point.append(list(point))

    else:
        print("d error")

    # print(total_point)
    pointSrc = np.float32(total_point)  # 原始图像中 4点坐标
    pointDst = np.float32([[[0, 0], [500, 0], [0, 500], [500, 500]]])  # 变换图像中 4点坐标
    MP = cv2.getPerspectiveTransform(pointSrc, pointDst)  # 计算投影变换矩阵 M
    imgP = cv2.warpPerspective(image_bin, MP, (500, 500))  # 用变换矩阵 M 进行投影变换

    # pointSrc = np.float32(total_point)  # 原始图像中 4点坐标
    # pointDst = np.float32([[[0, 0], [500, 0], [0, 500], [500, 500]]])  # 变换图像中 4点坐标
    # MP = cv2.getPerspectiveTransform(pointDst, pointSrc)  # 计算投影变换矩阵 M
    # re_imgP = cv2.warpPerspective(imgP, MP, (w, h))  # 用变换矩阵 M 进行投影变换
    # cv2.imwrite("re" + file_name, re_imgP)

    cv2.imwrite(end_image_root+"e\\" + file_name, imgP)
    cv2.imwrite(end_image_root+"c\\"+file_name, image_temp)
    return imgP, pointSrc, pointDst


if __name__ == '__main__':
    rootDir = "D:\\SYS_File\\Desktop\\Python\\image\\low\\"
    filename = '175446.jpg'
    template = img = np.zeros((500, 500), dtype=np.uint8)
    main_image = cv2.imread(rootDir + filename)
    h, w, _ = main_image.shape
    # 获取模板
    t = len(2 * os.listdir(rootDir))
    from tqdm import tqdm

    bar = tqdm(total=t)
    for filename in os.listdir(rootDir):
        name = filename
        pathname = os.path.join(rootDir, filename)
        img, pointSrc, pointDst = main(pathname, filename)
        img_row = img.shape[0]
        img_col = img.shape[1]

        # 利用遍历操作对二维灰度图作取反操作
        for i in range(img_row):
            for j in range(img_col):
                temp1 = img[i, j]
                temp2 = template[i, j]
                if temp1 or temp2:
                    template[i, j] = 255
                else:
                    template[i, j] = 0
        bar.update(1)
    # # 比较
    damaged_point = []
    for filename in os.listdir(rootDir):
        name = filename
        end_image = cv2.imread(rootDir + filename)
        draw_image = end_image.copy()
        # 图像二值化
        image_bin_temp = end_image.copy()
        image_bin_temp = cv2.cvtColor(image_bin_temp, cv2.COLOR_BGR2GRAY)
        # 边缘缺失
        ret2, image_bin = cv2.threshold(image_bin_temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 边缘破损

        ret, dam_img_bin = cv2.threshold(src=image_bin_temp, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

        pathname = os.path.join(rootDir, filename)
        img, pointSrc, pointDst = main(pathname, filename)  # 这里主要是反映射坐标求

        # 把模板反映射
        MP = cv2.getPerspectiveTransform(pointDst, pointSrc)  # 计算投影变换矩阵 M
        end_template = cv2.warpPerspective(template, MP, (w, h))  # 用变换矩阵 M 进行投影变换
        cv2.imwrite(end_image_root+"re\\" + filename, end_template)

        img_row = image_bin.shape[0]
        img_col = image_bin.shape[1]

        cv2.imwrite(end_image_root+"bin\\" + filename, image_bin)
        # 利用遍历操作对二维灰度图作取反操作
        for i in range(img_row):
            for j in range(img_col):
                if end_template[i, j] == 255 and image_bin[i, j] == 0:
                    damaged_point.append([i, j])

                if dam_img_bin[i, j] == 255 and end_template[i, j] == 255:
                    damaged_point.append([i, j])

        for point in damaged_point:
            draw_image[point[0], point[1]] = [0, 0, 255]
        aa = len(damaged_point)
        cv2.imwrite(end_image_root+"end\\" + str(aa) + "_" + filename, draw_image)
        damaged_point.clear()
        bar.update(1)
