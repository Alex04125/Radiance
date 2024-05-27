from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Numbers(BaseModel):
    a: float
    b: float

def calculate_operations(a, b):
    sum_result = a + b
    difference_result = a - b
    product_result = a * b
    division_result = a / b if b != 0 else "Division by zero error"
    return sum_result, difference_result, product_result, division_result

@app.post("/calculate")
async def calculate(numbers: Numbers):
    try:
        sum_result, difference_result, product_result, division_result = calculate_operations(numbers.a, numbers.b)
        response = {
            "sum": sum_result,
            "difference": difference_result,
            "product": product_result,
            "division": division_result
        }
        return response
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
