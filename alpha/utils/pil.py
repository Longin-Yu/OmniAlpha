import numpy as np
from PIL import Image
from typing import List, Optional, Union, Literal

BACKGROUND_TYPE = Union[Image.Image, float, Literal["checkerboard", "white", "black"]]
IMAGE_GROUP_TYPE = List[Image.Image]
IMAGE_ROW_TYPE = List[IMAGE_GROUP_TYPE]

def alpha_blend(foreground: Image.Image, background: BACKGROUND_TYPE) -> Image.Image:
    """
    将带 Alpha 通道的图像叠加到指定的背景上。
    """
    # 统一转为 RGBA 确保合成逻辑一致
    fg = foreground.convert("RGBA")
    width, height = fg.size

    # 处理背景类型
    if isinstance(background, Image.Image):
        bg = background.convert("RGBA").resize((width, height))
    elif background == "white":
        bg = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    elif background == "black":
        bg = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    elif background == "checkerboard":
        # 创建一个简单的棋盘格背景
        grid_size = 20
        bg_array = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                color = 200 if (x // grid_size + y // grid_size) % 2 == 0 else 255
                bg_array[y:y+grid_size, x:x+grid_size] = [color, color, color, 255]
        bg = Image.fromarray(bg_array)
    elif isinstance(background, (int, float)):
        # 灰色数值 (0-255 或 0-1)
        val = int(background) if background > 1 else int(background * 255)
        bg = Image.new("RGBA", (width, height), (val, val, val, 255))
    else:
        raise ValueError(f"Unsupported background type: {background}")

    # 使用 Alpha 混合
    return Image.alpha_composite(bg, fg)


def concat_image(
    *images: Union[Image.Image, List[Image.Image]],
    concat_on_row: bool = True,
    gap: int = 0,
) -> Image.Image:
    """
    拼接多张图像。
    concat_on_row=True:  水平拼接 (Row)，高度取 Max，宽度累加。
    concat_on_row=False: 垂直拼接 (Column)，宽度取 Max，高度累加。
    其余位置保持透明。
    """
    if not images:
        raise ValueError("Image list is empty")
    
    all_images = []
    for img in images:
        if isinstance(img, list):
            all_images.extend(img)
        else:
            all_images.append(img)
    images = all_images

    # 统一转为 RGBA
    imgs = [img.convert("RGBA") for img in images]
    widths, heights = zip(*(i.size for i in imgs))

    if concat_on_row:
        # 水平排列：总宽为和，高取最大
        dst_w = sum(widths) + gap * (len(imgs) - 1)
        dst_h = max(heights)
    else:
        # 垂直排列：总高为和，宽取最大
        dst_w = max(widths)
        dst_h = sum(heights) + gap * (len(imgs) - 1)

    # 创建透明底图 (0,0,0,0)
    canvas = Image.new("RGBA", (dst_w, dst_h), (0, 0, 0, 0))

    current_pos = 0
    for img in imgs:
        if concat_on_row:
            canvas.paste(img, (current_pos, 0))
            current_pos += img.width + gap
        else:
            canvas.paste(img, (0, current_pos))
            current_pos += img.height + gap

    return canvas


def create_image_grid(
    rows: List[IMAGE_ROW_TYPE],
    gap: int = 10,
    group_gap: Optional[int] = None,
) -> Image.Image:
    """
    把若干行 RGBA 图像拼成一个双背景网格：
    左侧白底，右侧黑底。
    每一行支持可变长度。
    推荐输入结构为 List[List[List[Image]]]:
        rows -> groups -> images
    同组内使用 gap，不同组之间使用 group_gap。
    """
    if not rows:
        raise ValueError("rows must not be empty")
    if group_gap is None:
        group_gap = gap * 3

    row_canvases = []
    for row in rows:
        if not row:
            continue

        group_canvases = []
        for group in row:
            if not group:
                continue
            group_canvases.append(concat_image(group, concat_on_row=True, gap=gap))

        if not group_canvases:
            continue

        row_canvases.append(concat_image(group_canvases, concat_on_row=True, gap=group_gap))

    if not row_canvases:
        raise ValueError("rows must contain at least one image")

    rgba_grid = concat_image(row_canvases, concat_on_row=False, gap=gap)
    white_grid = alpha_blend(rgba_grid, "white").convert("RGB")
    black_grid = alpha_blend(rgba_grid, "black").convert("RGB")
    return concat_image([white_grid, black_grid], concat_on_row=True, gap=gap).convert("RGB")

def resize_image_to_max_pixels(image: Optional[Image.Image], max_pixels: Optional[int]) -> Optional[Image.Image]:
    if image is None or max_pixels is None:
        return image

    total_pixels = image.width * image.height
    if total_pixels <= max_pixels:
        return image

    ratio = (max_pixels / total_pixels) ** 0.5
    new_width = max(1, int(image.width * ratio))
    new_height = max(1, int(image.height * ratio))
    return image.resize((new_width, new_height), Image.LANCZOS)


def concat_cross_matrix(
    foregrounds: List[Image.Image], 
    backgrounds: List[BACKGROUND_TYPE], 
    foreground_on_row: bool = False
) -> Image.Image:
    """
    创建一个交叉矩阵。
    如果 foreground_on_row=True: 
        每一行显示同一个 foreground，每一列显示同一个 background。
    如果 foreground_on_row=False (默认): 
        每一行显示同一个 background，每一列显示同一个 foreground。
    """
    matrix_rows = []

    if not foreground_on_row:
        # 外层循环背景 (Row)，内层循环前景 (Col)
        for bg in backgrounds:
            row_images = [alpha_blend(fg, bg) for fg in foregrounds]
            # 水平拼接这一行
            matrix_rows.append(concat_image(row_images, concat_on_row=True))
        # 将所有行垂直拼接
        return concat_image(matrix_rows, concat_on_row=False)
    
    else:
        # 外层循环前景 (Row)，内层循环背景 (Col)
        for fg in foregrounds:
            row_images = [alpha_blend(fg, bg) for bg in backgrounds]
            matrix_rows.append(concat_image(row_images, concat_on_row=True))
        return concat_image(matrix_rows, concat_on_row=False)
