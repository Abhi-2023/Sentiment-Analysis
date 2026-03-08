from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.predict import predict_sentiment


app = FastAPI()

templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse(
        'index.html',
        {'request':request, 'result':None }
    )
    
@app.post('/predict', response_class=HTMLResponse)
def predict(request: Request, review: str=Form(...)):
    sentiment = predict_sentiment(review=review)
    
    return templates.TemplateResponse(
        'index.html',
        {
            'request':request,
            'result':sentiment,
            'review':review
        }
    )