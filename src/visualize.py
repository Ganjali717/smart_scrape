import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import numpy as np

# Импортируем пайплайн
from src.pipeline import SmartScrapePipeline


def visualize_result(url, result, output_file="result_viz.png"):
    # Создаем белое полотно размером с экран (примерно)
    # В идеале нужно скачать скриншот из FitLayout, но пока нарисуем схему

    if result is None:
        return

    # Создаем белое полотно размером с экран (примерно)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Рисуем "страницу" (координаты из FitLayout обычно большие)
    # Инвертируем Y, так как в вебе 0 сверху
    ax.set_ylim(1200, 0)
    ax.set_xlim(0, 1280)
    ax.set_title(f"SmartScrape Extraction Result\n{url}", fontsize=10)

    # Цвета для классов
    colors = {"price": "green", "title": "red", "other": "gray"}

    print("\n[Viz] Drawing Bounding Boxes...")

    for label, data in result.items():
        bbox = data["bbox"]  # [x, y, w, h]
        confidence = data["confidence"]
        text = data["text"][:30] + "..."  # Обрезаем для красоты

        x, y, w, h = bbox

        # Рисуем прямоугольник
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor=colors.get(label, "blue"),
            facecolor="none",
        )
        ax.add_patch(rect)

        # Добавляем подпись
        plt.text(
            x,
            y - 5,
            f"{label.upper()} ({confidence:.2f})",
            color=colors.get(label, "blue"),
            fontsize=9,
            weight="bold",
        )
        plt.text(x, y + h + 15, text, color="black", fontsize=8, style="italic")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(output_file)
    print(f"✅ Visualization saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    url = "https://books.toscrape.com/catalogue/forever-and-forever-the-courtship-of-henry-longfellow-and-fanny-appleton_894/index.html"

    # Запускаем пайплайн снова
    pipeline = SmartScrapePipeline()
    final_data = pipeline.run(url)

    # Рисуем
    visualize_result(url, final_data)
