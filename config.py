"""
Configuration file for SmartScrape.
Centralizes all environment variables and service identifiers.
"""

# --- FITLAYOUT CONNECTION SETTINGS ---
# Базовый URL API. Проверь порт (обычно 8080)
API_BASE_URL = "https://layout.fit.vutbr.cz/api"

# Имя твоего репозитория, которое мы видели на скриншоте
TARGET_REPOSITORY_NAME = "IE AI"

# --- SERVICE IDENTIFIERS ---
# ID сервисов могут меняться. Если код упадет, мы проверим их через getServiceList.
# Обычно для Puppeteer ID именно такой:
SERVICE_RENDER_ID = "FitLayout.Puppeteer"

# Для сегментации (разбиения на блоки). VIPS - самый надежный вариант.
SERVICE_SEGMENTATION_ID = "FitLayout.VIPS"

SERVICE_AUTH_TOKEN = "Bearer eyJraWQiOiJcL3ByaXZhdGVLZXkucGVtIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnYW5qYWxpNzE3IiwidXBuIjoiZ2FuamFsaTcxNyIsImF1dGhfdGltZSI6MTc2MzkxNzQzNCwiaXNzIjoiand0YXV0aHNlcnYiLCJncm91cHMiOlsidXNlciJdLCJleHAiOjE3NjM5MjQ2MzQsImlhdCI6MTc2MzkxNzQzNCwianRpIjoiMjUiLCJlbWFpbCI6ImdhbmphbGkuaW1hbm92QGdtYWlsLmNvbSJ9.Ui2d6mGE335CjNmQOj5cJ5AuIp4wW0D-m6Z7pdAck1JDlBGATO0o7Ua2KL1kE5ug8Q9Qqp9S3oHAvijgw14MhY18dpb55XjDBoPI_gbprFLxi7wJugQAECM5l8AxlEnm42m3u3zZXutguLawdUVN7ADWZZGz37RDxEDCVVBQge1EEYyWKtzmOcGlEkIEiNyKzwtrKbTGxy5OcGIONddUxLR5Wa2SGzn-ccYH46CTZixUA5fM0DfwovnkkBi70x9aa8mf6Rs5_biEEmUHdCjHeNHpuutOLg77kU8JSgPlyensDLEOYnjch7RkJTn9neb3O17h5lakAzRPVf2emWqkog"

# --- ML SETTINGS (Пригодятся позже) ---
# Размерность векторов для нейросети
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
