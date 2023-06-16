class OCREngine:
    def get_data(self, image_path: str) -> list:
        raise NotImplementedError

    def get_text(self, image_path: str) -> str:
        raise NotImplementedError

    def get_boxes(self, image_path: str) -> list:
        raise NotImplementedError

    def get_text_with_prob(self, image_path: str) -> list:
        raise NotImplementedError

    def is_text_in_image(self, image_path, text):
        texts = self.get_text(image_path)
        return sum([1 if text in t else 0 for t in texts]) > 0
