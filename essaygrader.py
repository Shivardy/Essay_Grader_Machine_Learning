from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import nltk
import pickle
import os
import threading
import numpy    
from enchant.tokenize import EmailFilter,URLFilter
from enchant.checker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import language_check
from bs4 import BeautifulSoup 
import urllib.request
from imp import reload
from sklearn import tree
import sys
from google import search
import tkinter as tk
from tkinter import ttk
reload(sys)
def featureextraction(example,self):

    features=[]
    topic=example.split('\n')
    topic=topic[0]
    self.d1=ttk.Label(self.display ,text="Essay Title")
    self.d1.pack()
    self.d2=ttk.Label(self.display,wraplength = 480,anchor="center", text=topic)
    self.d2.pack()
    sentexample=sent_tokenize(example)
    no_of_sentences= len(sentexample)
    features.append(no_of_sentences)
    self.d3=ttk.Label(self.display,text="No of sentences are %s" % (no_of_sentences))
    self.d3.pack()
    avgsentlen=[]
    for i in sentexample:
        avgsentlen.append(len(word_tokenize(i)))
    avgsentlen= int(numpy.average(avgsentlen))
    features.append(avgsentlen)
    self.d4=ttk.Label(self.display,text="Average sentences length is %s" % (avgsentlen))
    self.d4.pack()
    wordexample=word_tokenize(example)
    stop_words=stopwords.words("english")
    stop_words.append(",")
    stop_words.append("?")
    stop_words.append(".")
    filteredwords=[]
    for w in wordexample:
        if w not in stop_words:
            w.lower()
            filteredwords.append(w)
    no_of_words=len(filteredwords)
    features.append(no_of_words)
    self.d5=ttk.Label(self.display,text="No of words after filtering stops words are %s" % (no_of_words))
    self.d5.pack()
    temp=len(set(filteredwords))
    features.append(temp)
    self.d6=ttk.Label(self.display,text="No of unique words are %s " % (temp))
    self.d6.pack()
    temp=len(set(filteredwords))
    lexicaldiveristy=numpy.true_divide(temp,no_of_words)
    features.append(lexicaldiveristy)
    self.d7=ttk.Label(self.display,text="Lexical Diversity is %s" % (lexicaldiveristy))
    self.d7.pack()
    tagged=sorted(nltk.pos_tag(filteredwords))
    nouns=[]
    adjectives=[]
    adverbs=[]
    verbs=[]
    for i in  tagged:
        if i[1]=='NNP' or i[1]=='NN' or i[1]=='NNS'or i[1]=='NNPS':
            nouns.append(i[0].lower())
    noundensity=numpy.true_divide(len(nouns),len(tagged))
    features.append(noundensity)
    self.d8=ttk.Label(self.display,text="Noun density is %s" % (noundensity))
    self.d8.pack()
    for i in tagged:
        if i[1]=='RB' or i[1]=='RBR' or i[1]=='RBS':
            adverbs.append(i[0])
    adverbdensity=numpy.true_divide(len(adverbs),len(tagged))
    features.append(adverbdensity)
    self.d9=ttk.Label(self.display,text="Adverb density is %s" % (adverbdensity))
    self.d9.pack()
    for i in tagged:
        if i[1]=='JJ' or i[1]=='JJR' or i[1]=='JJS': 
            adjectives.append(i[0])
    adjectivedensity=numpy.true_divide(len(adjectives),len(tagged))
    features.append(adjectivedensity)
    self.d10=ttk.Label(self.display,text="Adjective density is %s" % (adjectivedensity))
    self.d10.pack()
    for i in tagged:
        if i[1]=='VB' or i[1]=='VBD' or i[1]=='VBG'or i[1]=='VBN' or i[1]=='VBP' or i[1]=='VBZ':
            verbs.append(i[0])
    verbdensity=numpy.true_divide(len(verbs),len(tagged))
    features.append(verbdensity)
    self.d11=ttk.Label(self.display,text="Verb density is %s" % (verbdensity))
    self.d11.pack()
    s=SpellChecker("en_US",filters=[EmailFilter,URLFilter])
    s.set_text(example)
    spellerror=[]
    for error in s:
        spellerror.append(error.word)
    noofspell=len(spellerror)
    features.append(len(spellerror))
    
    self.d12=ttk.Label(self.display,wraplength = 450,text="There are %s Spelling mistakes, They are %s" % (noofspell,spellerror))
    self.d12.pack()
    spellerrorrate=numpy.true_divide(len(spellerror),len(wordexample))
    features.append(spellerrorrate)
    self.d13=ttk.Label(self.display,text="Spell Error rate is %s" % (spellerrorrate))
    self.d13.pack()
    grammar=language_check.LanguageTool('en-US')
    grammarerrors=grammar.check(example)
    features.append(len(grammarerrors))
    grammarmistakes=len(grammarerrors)
    self.d14=ttk.Label(self.display,text="No of grammatical mistakes are %s" % (grammarmistakes))
    self.d14.pack()
    linkarray=search(topic,stop=10)
    for link in linkarray:
        html = urllib.request.urlopen(link)
        soup = BeautifulSoup(html,"html.parser")
        a=""
        for para in soup.find_all("p"):
            a=a+para.text
        if len(word_tokenize(a))> 350:
            break
    a=word_tokenize(a)
    a=set(a)
    a=nltk.pos_tag(a)
    nounwords=[]
    for i in  a:
        if i[1]=='NNP' or i[1]=='NN' or i[1]=='NNS'or i[1]=='NNPS':
            nounwords.append(i[0])
    for i in nounwords:
        if "." in i:
            nounwords.remove(i)
    synonyms=[]
    for i in nounwords:
        for syn in wordnet.synsets(i):
            for l in syn.lemmas():
                if "_" not in l.name():
                    synonyms.append(l.name().lower())
    synonyms=synonyms+nounwords
    synonyms=set(synonyms)
    matchednouns=[]
    for i in nouns:
        if i in synonyms:
            matchednouns.append(i)
    essayrelevancy=numpy.true_divide(len(matchednouns),len(nouns))
    features.append(essayrelevancy)
    self.d16=ttk.Label(self.display,text="The Essay Relevancy is %s" % (essayrelevancy))
    self.d16.pack()
    
    input=open("essaymodel.pkl","rb")
    clf=pickle.load(input)
    input.close()
    predict=clf.predict([features])
    score=predict[0]
    self.d17=ttk.Label(self.display,wraplength=580,text="______________________________________________________________________")
    self.d17.pack()
    self.bytes=200
    self.d18=ttk.Label(self.display,style='Score.TLabel',text="The Essay Score is %s" % (score))
    self.d18.pack()
    self.retrybutton=ttk.Button(self.display,text="Retry Another Essay !",command= self.retry)
    self.retrybutton.pack(pady=15)


def trainfeatureextraction(example,self):
    features=[]
    topic=example.split('\n')
    score=topic[0]
    topic=topic[1]
    features.append(score)
    sentexample=sent_tokenize(example)
    no_of_sentences= len(sentexample)
    features.append(no_of_sentences)
    avgsentlen=[]
    for i in sentexample:
        avgsentlen.append(len(word_tokenize(i)))
    avgsentlen= int(numpy.average(avgsentlen))
    features.append(avgsentlen)
    #print "average sentence lenght is ", avgsentlen
    wordexample=word_tokenize(example)
    stop_words=stopwords.words("english")
    stop_words.append(",")
    stop_words.append("?")
    stop_words.append(".")
    filteredwords=[]
    for w in wordexample:
        if w not in stop_words:
            w.lower()
            filteredwords.append(w)
    no_of_words=len(filteredwords)
    features.append(no_of_words)
    #print "no of words after filtering stopwords are ", no_of_words
    features.append(len(set(filteredwords)))
    #print "no of unique words are ",len(set(filteredwords))
    temp=len(set(filteredwords))
    lexicaldiveristy=numpy.true_divide(temp,no_of_words)
    features.append(lexicaldiveristy)
    #print "lexical diveristy is ", lexicaldiveristy
    tagged=sorted(nltk.pos_tag(filteredwords))
    nouns=[]
    adjectives=[]
    adverbs=[]
    verbs=[]
    for i in  tagged:
        if i[1]=='NNP' or i[1]=='NN' or i[1]=='NNS'or i[1]=='NNPS':
            nouns.append(i[0].lower())
    noundensity=numpy.true_divide(len(nouns),len(tagged))
    features.append(noundensity)
    #print "uniqe nouns density is ",noundensity
    for i in tagged:
        if i[1]=='RB' or i[1]=='RBR' or i[1]=='RBS':
            adverbs.append(i[0])
    adverbdensity=numpy.true_divide(len(adverbs),len(tagged))
    features.append(adverbdensity)
    #print "unique adverbs density is ",adverbdensity
    for i in tagged:
        if i[1]=='JJ' or i[1]=='JJR' or i[1]=='JJS': 
            adjectives.append(i[0])
    adjectivedensity=numpy.true_divide(len(adjectives),len(tagged))
    features.append(adjectivedensity)
    #print "unique adjectives density is ",adjectivedensity
    for i in tagged:
        if i[1]=='VB' or i[1]=='VBD' or i[1]=='VBG'or i[1]=='VBN' or i[1]=='VBP' or i[1]=='VBZ':
            verbs.append(i[0])
    verbdensity=numpy.true_divide(len(verbs),len(tagged))
    features.append(verbdensity)
    #print "unique verbs density  is ",verbdensity
    s=SpellChecker("en_US",filters=[EmailFilter,URLFilter])
    s.set_text(example)
    spellerror=[]
    for error in s:
        spellerror.append(error.word)
    features.append(len(spellerror))
    features.append(spellerror)
    #print "There are ",len(spellerror)," spelling mistakes, They are : ", spellerror
    spellerrorrate=numpy.true_divide(len(spellerror),len(wordexample))
    features.append(spellerrorrate)
    #print "The spell error rate is ",spellerrorrate
    grammar=language_check.LanguageTool('en-US')
    grammarerrors=grammar.check(example)
    features.append(len(grammarerrors))
    #print "number of gramatical mistakes ",len(grammarerrors) 
    linkarray=search(topic,stop=10)
    for link in linkarray:
        html = urllib.request.urlopen(link)
        soup = BeautifulSoup(html,"html.parser")
        a=""
        for para in soup.find_all("p"):
            a=a+para.text
        if len(word_tokenize(a))> 350:
            break
    a=word_tokenize(a)
    a=set(a)
    a=nltk.pos_tag(a)
    nounwords=[]
    for i in  a:
        if i[1]=='NNP' or i[1]=='NN' or i[1]=='NNS'or i[1]=='NNPS':
            nounwords.append(i[0])
    for i in nounwords:
        if "." in i:
            nounwords.remove(i)
    synonyms=[]
    for i in nounwords:
        for syn in wordnet.synsets(i):
            for l in syn.lemmas():
                if "_" not in l.name():
                    synonyms.append(l.name().lower())
    synonyms=synonyms+nounwords
    synonyms=set(synonyms)
    matchednouns=[]
    for i in nouns:
        if i in synonyms:
            matchednouns.append(i)
    essayrelevancy=numpy.true_divide(len(matchednouns),len(nouns))
    features.append(essayrelevancy)
    #print "the essay relevancy is ",essayrelevancy
    return features

def trainfunction(path,self):
    features=[]
    for file in os.listdir(path):
        if file.endswith(".txt"):
            example=open(os.path.join(path,file))
            e=example.read()
            f=trainfeatureextraction(e,self)
            features.append(f)
    scores=[]

    for eachessay in features:
        scores.append(eachessay[0])
        eachessay.remove(eachessay[0])
        eachessay.remove(eachessay[10])
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(features,scores)
    output=open("essaymodel.pkl",'wb')
    pickle.dump(clf,output)
    output.close()
    self.a=self.count
    self.traindisplay.destroy()
    ttk.Label(self.traincomplete, text = 'Training has been successfully Completed!, Hit End Training',style='Subhead.TLabel').pack(pady=5)
    self.endtrain.state(['!disabled'])
    

class SampleApp(tk.Tk):

    def __init__(self, *args,**kwargs):
        tk.Tk.__init__(self,*args, **kwargs)
        self.title("Essay Grader Major Project -NMREC ( 67,A5,B3,B7 )")
        self.resizable(False,False)
        self.configure(background='#ECECEC')
        self.style=ttk.Style()
        self.style.configure('TLabel',font=('Arial',14))
        self.style.configure('Header.TLabel',font=('Arial',24,'bold'))
        self.style.configure('Subhead.TLabel',font=('Arial',18,))
        self.style.configure('Score.TLabel',font=('Arial',20,'bold'))



        self.frame_header=ttk.Frame(self)
        self.frame_header.pack()
        ttk.Label(self.frame_header,style='Header.TLabel',text="Essay Grader with Machine Learning!").grid(row=0,column=0)
        ttk.Label(self.frame_header, wraplength = 300, style='Subhead.TLabel',
                  text = "Select the file and hit submit..").grid(row = 1, column = 0)
        self.mainframe=ttk.Frame(self)
        self.mainframe.pack()
        self.frame_content = ttk.Frame(self.mainframe)
        self.frame_content.pack(pady=15)
        ttk.Label(self.frame_content, text = 'Essaypath:').grid(row = 0, column = 0, padx = 5)
        self.entry_name = ttk.Entry(self.frame_content, width = 50)
        self.entry_name.grid(row = 0, column = 1, padx = 5)
        self.testinfo=ttk.Button(self.frame_content, text = 'Info', command = self.testinfo,width=2.5)
        self.testinfo.grid(row=0,column=2)
        self.frame_button=ttk.Frame(self.mainframe)
        self.frame_button.pack()
        self.choosebutton=ttk.Button(self.frame_button,text="Choose..",command=self.choose)
        self.choosebutton.grid(row=0,column=0,sticky='e')
        self.submitbutton=ttk.Button(self.frame_button, text = 'Submit',command = self.submit)
        self.submitbutton.grid(row = 0, column = 1,sticky='w')
        self.progress=ttk.Frame(self)
        self.progress.pack()
        self.display=ttk.Frame(self)
        self.display.pack()
        self.endbutton=ttk.Frame(self)
        self.endbutton.pack(pady=10)
        self.trainbutton=ttk.Button(self.endbutton, command=self.train, text = 'TrainModel',width=16)
        self.trainbutton.pack()


    def retry(self):
        self.entry_name.state(['!disabled'])
        self.entry_name.delete(0,END)
        self.choosebutton.state(['!disabled'])
        self.submitbutton.state(['!disabled'])
        self.progress1.pack_forget()
        self.display.pack_forget()
        self.d1.destroy()
        self.d2.destroy()
        self.d3.destroy()
        self.d4.destroy()
        self.d5.destroy()
        self.d6.destroy()
        self.d7.destroy()
        self.d8.destroy()
        self.d9.destroy()
        self.d10.destroy()
        self.d11.destroy()
        self.d12.destroy()
        self.d13.destroy()
        self.d14.destroy()
        self.d16.destroy()
        self.d17.destroy()
        self.d18.destroy()
            
        self.retrybutton.destroy()

        self.trainbutton.pack()
        
    def testinfo(self):
        messagebox.showinfo(title="Testing Info", message='The first line of the essay must contain the essay title\n\n Followed by the essay content from the second line')

    def traininformation(self):
        messagebox.showinfo(title="Testing Info", message='The first line of the essay file must containt its score\n\n Second line must contain the essay title\n\n From the third line essay content must be present')

    def train(self):
        self.trainframe=ttk.Frame(self)
        self.trainframe.pack()
        ttk.Label(self.trainframe, text = 'Essay Folder:').grid(row = 0, column = 0, padx = 5)
        self.train_name = ttk.Entry(self.trainframe, width = 50)
        self.train_name.grid(row = 0, column = 1, padx = 5)
        self.traininfo=ttk.Button(self.trainframe, text = 'Info',width=2.5, command = self.traininformation)
        self.traininfo.grid(row=0,column=2)
        self.traincs=ttk.Frame(self)
        self.traincs.pack()
        self.trainchoose=ttk.Button(self.traincs,text="choose",command=self.choosetrain)
        self.trainchoose.grid(row=0,column=0,padx=5,sticky='e')
        self.trainsubmit=ttk.Button(self.traincs, text = 'Submit',command=self.trainsubmitted)
        self.trainsubmit.grid(row=0,column=1,sticky='w')
        self.entry_name.state(['disabled'])
        self.submitbutton.state(['disabled'])
        self.choosebutton.state(['disabled'])
        self.testinfo.state(['disabled'])
        self.trainbutton.state(['disabled'])
        self.traindisplay=ttk.Frame(self)
        self.traindisplay.pack()
        self.traincomplete=ttk.Frame(self)
        self.traincomplete.pack()
        self.endframe=ttk.Frame(self)
        self.endframe.pack()
        self.endtrain=ttk.Button(self.endframe,text="End Training",command=self.trainend,width=16)
        self.endtrain.grid(row=1,column=0,columnspan=2,pady=5)
        


    def trainend(self):
        self.trainframe.destroy()
        self.traincs.destroy()
        self.entry_name.state(['!disabled'])
        self.entry_name.delete(0,END)
        self.choosebutton.state(['!disabled'])
        self.submitbutton.state(['!disabled'])
        self.testinfo.state(['!disabled'])
        self.trainbutton.state(['!disabled'])
        self.traindisplay.destroy()
        self.endframe.destroy()
        self.traincomplete.destroy()

    def choose(self):
        file=filedialog.askopenfile()
        self.entry_name.delete(0,END)
        self.entry_name.insert(0,file.name)

    def choosetrain(self):
        file=filedialog.askdirectory()
        self.train_name.delete(0,END)
        self.train_name.insert(0,file)




    def proggressing(self):
        self.progress1=ttk.Progressbar(self.progress,orient=HORIZONTAL,length=200,mode="determinate",maximum=200,value=0)
        self.progress1.pack()
        self.bytes=0
        self.maxbytes=0
        self.read_bytes()


    def read_bytes(self):
        '''simulate reading 500 bytes; update progress bar'''
        self.bytes += 10
        self.progress1["value"] = self.bytes
        if self.bytes < 200:
            # read more bytes after 100 ms
            self.after(1000, self.read_bytes)
  



    def submit(self):
        
        path=self.entry_name.get()
        a=os.path.isfile(path)
        
        if(a and path.endswith(".txt")):
            self.display.pack()
            self.entry_name.state(['disabled'])
            self.submitbutton.state(['disabled'])
            self.choosebutton.state(['disabled'])
            self.trainbutton.pack_forget()
            example=open(path)
            example=example.read()
            t=threading.Thread(target=featureextraction,args=(example,self))
            t.start()
            s=threading.Thread(target=self.proggressing)
            s.start()
        else:
            messagebox.showwarning(title="warning", message='The file path you selected is invalid !\n\n Please select a valid file path')
        
    def trainprogressing(self):
        self.trainprogress.pack()
        self.a=1
        self.tprogress()

    def tprogress(self):
        self.a+=1
        self.trainprogress["value"]=self.a
        if self.a<self.count:
            self.after(500,self.tprogress)
        else:
            self.endtrain.state(['!disabled'])  
        


    def trainsubmitted(self):
        path=self.train_name.get()
        self.count=0
        for file in os.listdir(path):
            if file.endswith(".txt"):
                self.count=self.count+1
        if(self.count!=0 and path):
            
            if(path):
                
                self.train_name.state(['disabled'])
                self.trainsubmit.state(['disabled'])
                self.trainchoose.state(['disabled'])
                self.endtrain.state(['disabled'])
                self.count=self.count*20*2
                ttk.Label(self.traindisplay, text = 'Essay Grader Model Is Being Trained, Keep Waiting.',style='Subhead.TLabel').pack(pady=5)
                self.trainprogress=ttk.Progressbar(self.traindisplay,orient=HORIZONTAL,length=200,mode="determinate",maximum=self.count,value=0)

                t=threading.Thread(target=trainfunction,args=(path,self))
                t.start()
                s=threading.Thread(target=self.trainprogressing)
                s.start()
            
            else:
                messagebox.showwarning(title="warning", message='The folder path you selected is invalid !\n\n Please select a valid training folder path')
        else:
            messagebox.showwarning(title="warning", message='The folder path you selected is invalid !\n\n Please select a valid training folder path')            
        
        #self.trainbutton.state(['disabled'])
        

        
        



        




        


app = SampleApp()
app.mainloop()

