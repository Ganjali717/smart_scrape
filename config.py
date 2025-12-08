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

SERVICE_AUTH_TOKEN = "Bearer eyJraWQiOiJcL3ByaXZhdGVLZXkucGVtIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnYW5qYWxpNzE3IiwidXBuIjoiZ2FuamFsaTcxNyIsImF1dGhfdGltZSI6MTc2NTE4MzQ1MSwiaXNzIjoiand0YXV0aHNlcnYiLCJncm91cHMiOlsidXNlciJdLCJleHAiOjE3NjUxOTA2NTEsImlhdCI6MTc2NTE4MzQ1MSwianRpIjoiMjUiLCJlbWFpbCI6ImdhbmphbGkuaW1hbm92QGdtYWlsLmNvbSJ9.ElgTMsjSmNR_mWbjJkiltA0XFfjwvqIttAoHUBVYJLVdrOU5Un2SWJZVawxMMUOi5MfFqwwGdY9UARL8Oc3XfqIfcWNlyzoFcMjp3tuL-fkF9y4QgC6ZuA2rEO-yV83fF2XWxCOhWZMPT4K-Ndjzzht325AWibIZz6RTXGKW6441L7N0AXFnSSK1DOb6b6rAjCxG4Us__UCogSWxjJ0gQP2I3yoNC9qMsotj9Sl6_2YM55d4x12-ALYbSgdAJX8ED580nSDamU5RCtETZhkPu90A19wT5eRGysdNPMluK-7cP16Fu7M6TAoI2W0duIhZD6QVkhxvCSIohUNGVNGX2g"

# --- ML SETTINGS (Пригодятся позже) ---
# Размерность векторов для нейросети
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
