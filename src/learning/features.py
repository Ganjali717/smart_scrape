from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModel


class FeatureEncoder:
    """
    Отвечает за превращение сырых данных (текст, координаты, теги)
    в векторные представления (Embeddings).
    """

    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        device: Optional[str] = None,
    ):
        print(f"[FeatureEncoder] Loading BERT model: {model_name}...")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)
        self.bert.eval()  # выключаем dropout

        # Размерность текста берём из конфига модели
        self.text_dim = self.bert.config.hidden_size  # для bert-tiny = 128

        # x, y, w, h
        self.visual_dim = 4

        self.tag_map = {
            "div": 0,
            "span": 1,
            "a": 2,
            "img": 3,
            "p": 4,
            "h1": 5,
            "h2": 6,
            "li": 7,
            "ul": 8,
            "table": 9,
            "form": 10,
            "input": 11,
            "button": 12,
        }
        self.tag_dim = len(self.tag_map) + 1  # +1 для 'other'

    def encode_text(self, text: str) -> torch.Tensor:
        """Превращает текст в вектор (BERT embedding)."""
        if not text or str(text).strip() == "":
            return torch.zeros(self.text_dim, dtype=torch.float32)

        inputs = self.tokenizer(
            str(text)[:64],
            return_tensors="pt",
            padding="max_length",
            max_length=10,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert(**inputs)
            # [batch, seq_len, hidden] -> [hidden]
            emb = outputs.last_hidden_state[0, 0]

        return emb.cpu()  # чтобы потом всё было на CPU

    def encode_visual(self, x, y, w, h, page_w, page_h) -> torch.Tensor:
        """Нормализует координаты относительно размера страницы (0..1)."""
        page_w = page_w if page_w > 0 else 1920.0
        page_h = page_h if page_h > 0 else 1080.0

        return torch.tensor(
            [x / page_w, y / page_h, w / page_w, h / page_h],
            dtype=torch.float32,
        )

    def encode_tag(self, tag_name: str) -> torch.Tensor:
        """One-Hot кодирование HTML тега."""
        idx = self.tag_map.get(tag_name.lower(), len(self.tag_map))
        vec = torch.zeros(self.tag_dim, dtype=torch.float32)
        vec[idx] = 1.0
        return vec

    def get_output_dim(self) -> int:
        return self.text_dim + self.visual_dim + self.tag_dim
