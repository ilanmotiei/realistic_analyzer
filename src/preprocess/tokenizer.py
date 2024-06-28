import json
from typing import List, Dict, Any

from src.config import Config


class Tokenizer:
    def __init__(self):
        with open(Config.ALL_ISRAELI_SETTLEMENTS_LOCATION, 'r', encoding='utf-8') as f:
            self.settlements_records: List[Dict[str, Any]] = json.load(fp=f)

        self.str_to_numeric_translator: Dict[str, int] = {
            self.normalize_settlement_name(settle_record.get('city_name_he')): settle_record.get('_id') for settle_record in self.settlements_records
        }
        self.numeric_to_str_translator: Dict[int, str] = {v: k for k, v in self.str_to_numeric_translator.items()}

    @staticmethod
    def normalize_settlement_name(name: str) -> str:
        return name.replace('(', '').replace(')', '').replace('-', ' ').strip()

    def settlement_str_to_numeric(self, settlement_str: str) -> int:
        return self.str_to_numeric_translator[settlement_str]

    def settlement_numeric_to_str(self, settlement_numeric: int) -> str:
        return self.numeric_to_str_translator[settlement_numeric]
