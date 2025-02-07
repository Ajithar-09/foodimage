from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import base64
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from fastapi.responses import JSONResponse

app = FastAPI()


chat_model = ChatOllama(model="llava:7b")


prompt = """
Identify the food item in the given image and provide details in **only** this format:

Food Image Name: [Exact name of the food item]
Calories: [Approximate calorie count per serving]
Protein: [Amount of protein per serving in grams]
Fat: [Amount of fat per serving in grams]

Do **not** include any explanation or extra text.
"""

@app.post("/analyze_image/")
async def analyze_image(file: UploadFile = File(...)):
    
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    
   
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    )

    
    response = chat_model.invoke([message])
    raw_response = response.content
    print("Raw Response:", raw_response)  
    
   
    match = re.search(
        r"Food Image Name:\s*(.*?)\s*(Calories:\s*[\d\-]+(?:\s*[\w]+)*)\s*(Protein:\s*[\d\-]+(?:\s*[\w]+)*)\s*(Fat:\s*[\d\-]+(?:\s*[\w]+)*)", 
        raw_response, re.DOTALL | re.IGNORECASE
    )
    
    result = {"error": "Could not analyze the image. Ensure the food item is clearly visible."}
    

    if match:
        food_name = match.group(1).strip().title()
        calories = match.group(2).strip() if match.group(2) else "Not available"
        protein = match.group(3).strip() if match.group(3) else "Not available"
        fat = match.group(4).strip() if match.group(4) else "Not available"


        result = {
            "Food Image Name": food_name,
            "Calories": calories + " per portion" if calories != "Not available" else calories,
            "Protein": protein + "g" if protein != "Not available" else protein,
            "Fat": fat + "g" if fat != "Not available" else fat
        }

    return JSONResponse(content={"raw_response": raw_response.strip(), "result": result})
