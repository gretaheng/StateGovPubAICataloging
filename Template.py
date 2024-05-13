from LanDic import dlan
from tika import parser
import pycountry
from pymarc import Record, Field, Subfield
from datetime import datetime
import yaml
import torch
from transformers import pipeline
from constants import APIKEY,organizationID
import openai
import pdfplumber
import csv

# Read the file
def read_pdf(pathpdf, pathurl):
    raw = parser.from_file(pathpdf)
    print('PDF Parser finds the metadata of this gov doc example: ',
          yaml.dump(raw['metadata'], default_flow_style=False))
    print("******************************************")
    rawcontent = raw['content'].replace("/n"," ")
    print("The first 1000 words of the gov doc example is: ", rawcontent[:1000])
    print("******************************************")
    fileurl = open(pathurl, 'r')
    url = fileurl.read()
    print("The url of the gov doc example is ", url)
    print("******************************************")

    dinfo = {}
    # Get language
    raw_lan = raw["metadata"]['Content-Language'].split("-")[0][:2]
    lan = pycountry.languages.get(alpha_2=raw_lan).name
    print("The language of the example file is:", lan)
    print("******************************************")
    dinfo["lan"] = dlan[lan]
    # Get titles
    titlea = [i for i in raw["metadata"]['dc:title'] if i != "Acrobat Accessibility Report"][0].capitalize()
    ind2245 = ""
    if titlea[:2].lower() == "a ":
        ind2245 = 2
    if titlea[:3].lower() == "an ":
        ind2245 = 3
    if titlea[:4].lower() == "the ":
        ind2245 = 4
    print("The main title of the file is:", titlea)
    print("******************************************")
    dinfo["title"] = titlea
    dinfo["titleind2"] = ind2245
    # publication year
    pubYear = raw["metadata"]['xmp:CreateDate'].split("T")[0].split("-")[0]
    print("The publication year is:", pubYear)
    print("******************************************")
    dinfo["year"] = pubYear
    # add url
    dinfo["url"] = url
    return dinfo

def marcbase(dinfo):
    # get marc 005
    currentDateAndTime = datetime.now()
    print("The record is created at:", currentDateAndTime)
    marc005 = str(currentDateAndTime).split(".")[0].replace("-", "").replace(":", "").replace(" ", "") + ".0"
    print('MARC 005 field is: ', marc005)
    print("******************************************")
    # get marc 006
    marc006 = "m     o  d s      "
    print('MARC 006 field is: ', marc006)
    print("******************************************")
    # get marc 007
    marc007 = "cr |n|||||||||"
    print('MARC 007 field is: ', marc007)
    print("******************************************")
    # get marc 008
    marc008 = marc005[2:8]
    if dinfo['year']:
        marc008 += "s" + dinfo['year'] + "    "
    else:
        marc008 += "nuuuuuuuu"
    marc008 += "cau" + "          f000 0 " + dinfo["lan"] + " " + "d"
    print('MARC 008 field is: ', marc008)
    print("******************************************")
    # create the MARC record
    record = Record()
    # 005 field
    record.add_field(Field(tag='005', data=marc005))

    # 006 field
    record.add_field(Field(tag='006', data=marc006))

    # 007 field
    record.add_field(Field(tag='007', data=marc007))

    # 008 field
    record.add_field(Field(tag='008', data=marc008))

    # 040 field
    record.add_field(
        Field(
            tag='040',
            indicators=[' ', ' '],
            subfields=[
                Subfield(code='a', value='CDS'),
                Subfield(code='b', value='eng'),
                Subfield(code='e', value='rda'),
                Subfield(code='c', value='CDS')
            ]))

    # 245 field
    if dinfo["titleind2"] != "":
        record.add_field(
            Field(
                tag='245',
                indicators=[' ', dinfo["titleind2"]],
                subfields=[Subfield(code='a', value=dinfo["title"])]))
    else:
        record.add_field(
            Field(
                tag='245',
                indicators=[' ', ' '],
                subfields= [Subfield(code='a', value=dinfo["title"])]))

    # 264 field
    record.add_field(
        Field(
            tag='264',
            indicators=[' ', "1"],
            subfields=[
                Subfield(code='a', value='[California]'),
                Subfield(code='b', value='[publisher not identified]'),
                Subfield(code='c', value=dinfo["year"])
            ]))

    # 3XX fields
    record.add_field(
        Field(
            tag='300',
            indicators=[' ', " "],
            subfields=[Subfield(code='a', value='1 online resource')]))

    record.add_field(
        Field(
            tag='336',
            indicators=[' ', " "],
            subfields=[
                Subfield(code='a', value='text'),
                Subfield(code='b', value='txt'),
                Subfield(code='c', value='rdacontent')
            ]))

    record.add_field(
        Field(
            tag='337',
            indicators=[' ', " "],
            subfields=[
                Subfield(code='a', value='computer'),
                Subfield(code='b', value='c'),
                Subfield(code='c', value='rdamedia')
            ]))

    record.add_field(
        Field(
            tag='338',
            indicators=[' ', " "],
            subfields=[
                Subfield(code='a', value='online resource'),
                Subfield(code='b', value='cr'),
                Subfield(code='c', value='rdacarrier')
            ]))
    # 856 field
    record.add_field(
        Field(
            tag='856',
            indicators=['4', "0"],
            subfields=[Subfield(code='u', value=dinfo["url"])]))
    print("Below is the MARC records before AI generating summary and keywords:")
    print(yaml.dump(record.as_dict(), default_flow_style=False))
    print("******************************************")
    #write marc
    with open('sample-1.mrc', 'wb') as out:
        out.write(record.as_marc())
    return record

def HF_model_summarize(filetext):
    hf_name = 'pszemraj/led-large-book-summary'
    summarizer = pipeline(
        "summarization",
        hf_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    result = summarizer(
        filetext,
        min_length=16,
        max_length=256,
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=3.5,
        num_beams=4,
        early_stopping=True,
    )
    return result

def HF_Summary_Pages(paperContent, pagesplit, dic):
    listpage = [paperContent[x:x + pagesplit] for x in range(0, len(paperContent), pagesplit)]
    for i in range(len(listpage)):
        key = str(i)
        keytext = ""
        for page in listpage[i]:
            text = page.extract_text()
            keytext+= text
        textsum = HF_model_summarize(keytext)[0]["summary_text"]
        dic[str(pagesplit)][key] = textsum
    return dic

def OpenAI_Summary_Pages(paperContent, pagesplit, dic):
    listpage = [paperContent[x:x + pagesplit] for x in range(0, len(paperContent), pagesplit)]
    tldr_tag = "\n tl;dr:"
    alltext = []
    for i in range(len(listpage)):
        key = str(i)
        keytext = ""
        for page in listpage[i]:
            text = page.extract_text()
            keytext+= text
        keytext += tldr_tag
        alltext.append(keytext)
        response = openai.Completion.create(engine="gpt-3.5-turbo-instruct", prompt=text, temperature=0.3,
                                            max_tokens=140, top_p=1, frequency_penalty=0, presence_penalty=0,
                                            stop=["\n"])
        if response["choices"][0]["text"]:
            dic[str(pagesplit)][key]= response["choices"][0]["text"]
    return dic

def writeSum2CSV(outputname, dic):
    with open(outputname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in dic.items():
            writer.writerow([key, value])
    csvfile.close()
    return

def getSummary_HF(filepath):
    """
    Use hugging face text summarization model: https://huggingface.co/pszemraj/led-large-book-summary generate
    1. page summary
    2. summarize every 5 pages
    3. summarize every 10 pages
    4. summarize every 15 pages
    5. summarize every 20 pages
    """
    page_summary_hf = {"1": {}, "5":{}, "10":{}, "15":{}, "20":{}}

    paperContent = pdfplumber.open(filepath).pages

    for page in paperContent:
        key = str(page).replace("<Page:","").replace(">","")
        text = page.extract_text()
        textsum = HF_model_summarize(text)[0]["summary_text"]
        page_summary_hf["1"][key] = textsum
    writeSum2CSV('HF1page.csv', page_summary_hf["1"])

    HF_Summary_Pages(paperContent, 5, page_summary_hf)
    writeSum2CSV('HF5page.csv', page_summary_hf["5"])

    HF_Summary_Pages(paperContent, 10, page_summary_hf)
    writeSum2CSV('HF10page.csv', page_summary_hf["10"])

    HF_Summary_Pages(paperContent, 15, page_summary_hf)
    writeSum2CSV('HF15page.csv', page_summary_hf["15"])

    HF_Summary_Pages(paperContent, 20, page_summary_hf)
    writeSum2CSV('HF20page.csv', page_summary_hf["20"])
    return


def getSummary_OpenAI(pathpdf, record):
    paperContent = pdfplumber.open(pathpdf).pages
    openai.api_key = APIKEY
    openai.organization = organizationID
    alltext = ""
    for page in paperContent:
        text = page.extract_text()
        alltext += text
    responsesum = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                   "content": "Write a clear and concise summary for the provided government publication. The summary should be less than 50 words:"},
                  {"role": "user",
                   "content": f"Write a summary of the following government publication:{alltext}\nDETAILED SUMMARY:"}], )
    if responsesum["choices"][0]["message"]["content"]:
        print("OpenAI API gives the summary following: ", responsesum["choices"][0]["message"]["content"])
    sum520 = responsesum["choices"][0]["message"]["content"]

    responsekw = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                   "content": "assign at most 5 keywords; do not number the keywords; use ',' to separate keywords; Do not add any words before the first subject heading or after the last subject headings; response should be all lowercase; no temporal keywords please, only topical keywords"},
            {"role": "user",
                   "content": f"Can you assign at most 5 keywords for the provided government documentation: {alltext}"}], )

    if responsekw["choices"][0]["message"]["content"]:
        print("OpenAI API gives the summary following: ", responsekw["choices"][0]["message"]["content"])
    kwlst = responsekw["choices"][0]["message"]["content"].split(", ")

    # 520 field
    record.add_field(
        Field(
            tag='520',
            indicators=[' ', ' '],
            subfields=[
                Subfield(code='a', value=sum520)
            ]))

    # 653 field
    for k in kwlst:
        record.add_field(
            Field(
                tag='653',
                indicators=[' ', ' '],
                subfields=[
                    Subfield(code='a', value=k.strip("."))
                ]))

    print("Below is the MARC records after AI generating summary and keywords:")
    print(yaml.dump(record.as_dict(), default_flow_style=False))
    print("******************************************")

    #write marc
    with open('sample-2.mrc', 'wb') as out:
        out.write(record.as_marc())

    return record

