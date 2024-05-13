Below are the codes used in the demo:

from Template import read_pdf, marcbase, getSummary_OpenAI


pathpdf = "GovDocExample/filepdf.pdf"


pathurl = "GovDocExample/fileurl.txt"


dinfo = read_pdf(pathpdf, pathurl)


record = marcbase(dinfo)


record = getSummary_OpenAI(pathpdf, record)
