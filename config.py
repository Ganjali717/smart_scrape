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

SERVICE_AUTH_TOKEN = "Bearer eyJraWQiOiJcL3ByaXZhdGVLZXkucGVtIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiJnYW5qYWxpNzE3IiwidXBuIjoiZ2FuamFsaTcxNyIsImF1dGhfdGltZSI6MTc2NTMyMjM2NSwiaXNzIjoiand0YXV0aHNlcnYiLCJncm91cHMiOlsidXNlciJdLCJleHAiOjE3NjUzMjk1NjUsImlhdCI6MTc2NTMyMjM2NSwianRpIjoiMjUiLCJlbWFpbCI6ImdhbmphbGkuaW1hbm92QGdtYWlsLmNvbSJ9.iL_clmyRHlMcI2sxEqml4l4gDEfC5mR3-lfbh3S1wiXQoLgmk6g4miaQHENv0kvPVN79km7NYG-ArsgNgI-Kbu8Cvt4r_qdyx7J8SwAVEnH_bO-iLRdWenD5otvMDNSS1Sz1rT33IzeUb8Ay-XSl6ySBri_iwtN6mhXktmM3UOIzuXt_E9uc-NJ3LTKqQNVCK1b_wIuyZX_-ve9Pk51zbhcBkFCtNyVz6medXqket9xtYBIodv6oWcX1J4zku3TJRI2hNsv02w-VyJnqbVAVLVWLYlFjD1lzNrS42JwDFZfmS_hhNAutbRZz1cGJFqqytWHhQooOjwZDmk_6C_SksA"

# --- ML SETTINGS (Пригодятся позже) ---
# Размерность векторов для нейросети
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
