import json
import math
import os
from json import JSONDecodeError
from multiprocessing import Pool
from typing import List, Dict, Any, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm


def create_prompt(floor_number: str) -> str:
    prompt: str = """
        Prompt: Given a string in Hebrew that describes various floors of a building, 
        such as "קומת קרקע, קומה ראשונה, קומת מרתף, גג", convert this information into a list of 
        numerical floor numbers. Floors that are special types, like basements or rooftops, 
        should be inferred based on their context and given appropriate numbers. 
        The output should be in the format {"floors": List[int], "sequencial": bool}, 
        where "floors" is the list of numerical floor numbers and "sequencial" is a boolean 
        indicating whether the floors set is a sequence of consecutive numbers.

        Input Example: "קומת קרקע, קומה ראשונה, קומת מרתף, גג"

        Expected Output: {"floors": [0, 1, -1, 2], "sequencial": true}

        Input Example 2: "קומת קרקע, קומה ראשונה, קומה שנייה, קומה שלישית"

        Expected Output 2: {"floors": [0, 1, 2, 3], "sequencial": true}

        Input Example: "קומת קרקע, קומה ראשונה, קומת מרתף, גג"

        Expected Output: {"floors": [0, 1, -1, 2], "sequencial": true}

        Input Example 3: "קומה ארבע-עשרה וקומה חמש עשרה + קומת גג"

        Expected Output 3: {"floors": [14, 15, 16], "sequencial": true}

        Input Example 4: "קומה ארבע-עשרה וקומה חמש עשרה + מרתף"

        Expected Output 4: {"floors": [14, 15, -1], "sequencial": false}

        Input Example 5: "קרקע+שניה+עליית גג"

        Expected Output 5: {"floors": [0, 2, 3], "sequencial": false}

        Input Example 6: "מרתף א' ו ב' קרקע ראשונה שניה"

        Expected Output 6: {"floors": [-1, 0, 1, 2], "sequencial": true}
    """
    prompt += floor_number
    prompt += f"""
        Your output should be simply the answer in the aforementioned format. 
        You shouldn't add anything else to your answer.
    """
    return prompt


client = OpenAI(
    api_key='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',
    organization='org-2gSXnlG2Pv2dEEFDiIgev2oS'
)

PAGE_SIZE: int = 100
NUM_CONCURRENT_REQUESTS: int = 10

with open('../../data/15062024/unique_floor_numbers_23062024.json', 'r', encoding='utf-8') as f:
    floors: List[str] = json.load(fp=f)


def get_floor_translation(floor: str) -> Optional[Dict[str, Any]]:
    prompt: str = create_prompt(floor)
    completion: ChatCompletion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    floor_formatted: str = completion.choices[0].message.content
    try:
        return json.loads(s=floor_formatted)
    except JSONDecodeError:
        return None


def worker(page_idx: int) -> None:
    result: Dict[str, Optional[Dict[str, Any]]] = {}
    for floor in tqdm(floors[page_idx * PAGE_SIZE: (page_idx + 1) * PAGE_SIZE], desc=f"Page index: {page_idx}"):
        floor_result: Optional[Dict[str, Any]] = get_floor_translation(floor)
        result.update({floor: floor_result})

    save_page(result, page_idx)


def get_page_filepath(idx: int) -> str:
    return f'floors_translation_pages/{idx}.json'


def save_page(result: Dict[str, Optional[Dict[str, Any]]], page_idx: int):
    with open(get_page_filepath(page_idx), 'w', encoding='utf-8') as fp:
        json.dump(obj=result, indent=2, fp=fp, ensure_ascii=False)


if __name__ == '__main__':
    page_indexes: List[int] = [idx for idx in range(math.ceil(len(floors) / PAGE_SIZE))]
    page_indexes_to_process: List[int] = [idx for idx in page_indexes if not os.path.isfile(get_page_filepath(idx))]
    print(f"Pages already processed: {len(page_indexes) - len(page_indexes_to_process)}/{len(page_indexes)}")
    with Pool(processes=NUM_CONCURRENT_REQUESTS) as pool:
        pool.starmap(worker, ((idx,) for idx in page_indexes_to_process))







