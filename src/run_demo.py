from src.pipeline import SmartScrapePipeline
import json


def main():
    pipeline = SmartScrapePipeline()

    # URL страницы товара (Books to Scrape)
    url = "https://books.toscrape.com/catalogue/forever-and-forever-the-courtship-of-henry-longfellow-and-fanny-appleton_894/index.html"

    result = pipeline.run(url)

    print("\n" + "=" * 40)
    print("🎯 SMARTSCRAPE FINAL OUTPUT")
    print("=" * 40)
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
