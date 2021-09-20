from django.shortcuts import render
#from .models import Summary
from .text_summarizer import summarization

# Create your views here.

def home(request):
    return render(request, 'summary/summary_form.html')

def textSum(request):
    if request.method=='POST':
        text=request.POST.get('content',None)
        result = summarization(text)

        return render(request,'summary/result.html',{'result':result})
