import json
import math
import os
from json import JSONDecodeError
from typing import List, Dict, Any, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm


PAGE_SIZE: int = 50


def get_page_filepath(idx: int) -> str:
    return f'/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data/floors_translations_pages_gpt4_batch/{idx}.json'


def save_page(result: Dict[str, Optional[Dict[str, Any]]], page_idx: int):
    with open(get_page_filepath(page_idx), 'w', encoding='utf-8') as f:
        json.dump(obj=result, indent=2, fp=f, ensure_ascii=False)


def read_content(idx: int) -> Any:
    filepath = get_page_filepath(idx)
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(fp=f)

    return None


def create_prompt(floors_str_list: List[str]) -> str:
    prompt: str = """
        Prompt: Given a string in Hebrew that describes various floors of a building, 
        such as "קומת קרקע, קומה ראשונה, קומת מרתף, גג", convert this information into a list of 
        numerical floor numbers. Floors that are special types, like basements or rooftops, 
        should be inferred based on their context and given appropriate numbers. 
        The output should be in the format {<input_floor>: {"floors": List[int], "sequential": bool}}, 
        where "floors" is the list of numerical floor numbers and "sequential" is a boolean 
        indicating whether the floors set is a sequence of consecutive numbers.
    
        Example Input: [
            ,"קומת קרקע, קומה ראשונה, קומת מרתף, גג"
            ,"קומת קרקע, קומה ראשונה, קומה שנייה, קומה שלישית"
            ,"קומה ארבע-עשרה וקומה חמש עשרה + קומת גג"
            ,"קומה ארבע-עשרה וקומה חמש עשרה + מרתף"
            ,"קרקע+שניה+עליית גג"
            "מרתף א' ו ב' קרקע ראשונה שניה"
        ]
        
        Example Output: {
            ,"קומת קרקע, קומה ראשונה, קומת מרתף, גג": {floors": [0, 1, -1, 2], "sequential": true"}
            ,"קומת קרקע, קומה ראשונה, קומה שנייה, קומה שלישית": {floors": [0, 1, 2, 3], "sequential": true"}
            ,"קומה ארבע-עשרה וקומה חמש עשרה + קומת גג": {floors": [14, 15, 16], "sequential": true"}
            ,"קומה ארבע-עשרה וקומה חמש עשרה + מרתף": {floors": [14, 15, -1], "sequential": false"}
            ,"קרקע+שניה+עליית גג": {"floors": [0, 2, 3], "sequential": false"}
            "מרתף א' ו ב' קרקע ראשונה שניה": {floors": [-1, 0, 1, 2], "sequential": true"}
        }
        Your input: 
    """
    prompt += str(floors_str_list)
    prompt += f"""
        (Your output should be simply the answer in the aforementioned format. 
        You shouldn't add anything else to your answer. no comments, NOTHING MORE)
        
        Output:
    """
    return prompt


def get_floors_translation(floors_str_list: List[str]) -> Optional[Dict[str, Any]]:
    prompt: str = create_prompt(floors_str_list)
    completion: ChatCompletion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    floors_formatted: str = completion.choices[0].message.content
    try:
        return json.loads(s=floors_formatted)
    except JSONDecodeError:
        try:
            return json.loads(floors_formatted.replace('```', '').replace('json', ''))
        except JSONDecodeError:
            return None


if __name__ == '__main__':
    client: OpenAI = OpenAI(
        api_key='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',  # not valid anymore friend..
        organization='org-2gSXnlG2Pv2dEEFDiIgev2oS'
    )
    with open('../../data/15062024/unique_floor_numbers_23062024.json', 'r', encoding='utf-8') as f:
        floors: List[str] = json.load(fp=f)

    floor_batches: List[List[str]] = [floors[i * PAGE_SIZE: (i+1) * PAGE_SIZE] for i in range(math.ceil(len(floors)/PAGE_SIZE))]
    for i, floor_batch in tqdm(enumerate(floor_batches), total=len(floor_batches)):
        if read_content(i) is not None:
            continue

        print(f"Page: {i}")
        prompt = create_prompt(floor_batch)
        # floor_result: Optional[Dict[str, Any]] = get_floors_translation(floor_batch)
        # save_page(floor_result, i)
