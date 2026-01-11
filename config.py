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

SERVICE_AUTH_TOKEN = "Bearer eyJraWQiOiJcL3ByaXZhdGVLZXkucGVtIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnYW5qYWxpNzE3IiwidXBuIjoiZ2FuamFsaTcxNyIsImF1dGhfdGltZSI6MTc2ODEzMjEyNCwiaXNzIjoiand0YXV0aHNlcnYiLCJncm91cHMiOlsidXNlciJdLCJleHAiOjE3NjgxMzkzMjQsImlhdCI6MTc2ODEzMjEyNCwianRpIjoiMjUiLCJlbWFpbCI6ImdhbmphbGkuaW1hbm92QGdtYWlsLmNvbSJ9.FcmoqyHFd--1yyyTIoWIJcBDnxm-xYlHC2wbiad3lT8fwByxz-5aJURrXcT4zA_ryekKblh8cPN0EImjdAkyPSMMvXkFGdzrthAP2pLsX5k6wOrvXUOSMnM6_L2X4Afred5MVEukl5gPUbGNZX_at4laH9SUf5YHWDSm2mIT-nVLcqJjoXyiGx8R-043dT77L8wYqqmWOYzMPFNnUs4WzGRGPuj-J7e08gk6kMedtGnHzKV1geZDotHftOUkx2YHBrHnWjdeOZtT3CN_yZaclXnC2---kQagaMznuPmm1srnDWjSoliezxHVtJH0_zbu0ZJxrEfNB5za0r7jCxA6hw"

# --- ML SETTINGS (Пригодятся позже) ---
# Размерность векторов для нейросети
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
