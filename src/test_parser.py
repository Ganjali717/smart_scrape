import json
from src.integration.fitlayout import FitLayoutClient
from src.learning.graph_builder import FitLayoutParser


def main():
    # 1. Получаем данные (как ты уже делал)
    client = FitLayoutClient()
    url = "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html"

    print("--- 1. Fetching Data ---")
    json_content = client.get_page_content(url)

    # 2. Парсим
    print("\n--- 2. Parsing & Building Graph ---")
    parser = FitLayoutParser()
    data, raw_nodes = parser.parse(json_content)

    if data:
        print("\n✅ SUCCESS!")
        print(f"Nodes created: {data.x.shape[0]}")
        print(f"Feature vector size: {data.x.shape[1]} (4 Visual + 128 Text + Tags)")
        print(f"Edges created: {data.edge_index.shape[1]}")

        print("\nПример сырого узла (для Solver):")
        print(raw_nodes[0])

        print("\nПример тензора (для GNN):")
        print(data.x[0][:10])  # Показываем первые 10 чисел
    else:
        print("❌ Parsing failed.")


if __name__ == "__main__":
    main()
