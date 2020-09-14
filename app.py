import os
from flask import Flask, render_template, request, send_file
from flask_mysqldb import MySQL

import PyPDF2
import re
import numpy as np
import nltk
import glob
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from fpdf import FPDF

nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import nltk
from nltk.util import ngrams
from difflib import SequenceMatcher
import datetime

path = "C:/xampp/htdocs/xampp/project-version9/"


class Text:
    def __init__(self, filename):
        self.filename = filename
        self.trigrams = self.ngrams(3)

    @property
    def text(self):
        return open(self.filename).read()

    def ngrams(self, n):
        tokenizer = nltk.RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
        self.spans = list(tokenizer.span_tokenize(self.text))
        tokens = tokenizer.tokenize(self.text)
        return list(ngrams(tokens, n))


class Matcher:
    def __init__(self, fileA, fileB, threshold, ngramSize):
        self.threshold = threshold
        self.ngramSize = ngramSize
        self.textA, self.textB = Text(fileA), Text(fileB)
        self.textAgrams = self.textA.ngrams(ngramSize)
        self.textBgrams = self.textB.ngrams(ngramSize)

    def match(self):
        sequence = SequenceMatcher(None, self.textAgrams, self.textBgrams)
        matchingBlocks = sequence.get_matching_blocks()
        highMatchingBlocks = [match for match in matchingBlocks if match.size > self.threshold]
        numBlocks = len(highMatchingBlocks)
        report.write('Number of sentences quoted = %s ' % numBlocks)
        report.write('\n\n\n')


global proname
global namefile


class Plagiarism:
    def __init__(self, filenumber, filename):
        self.filenumber = filenumber
        self.filename = filename

    def ReadPDFfile(self):
        pdf = open(path + "data/" + self.filename + '.pdf', 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdf)
        numPages = pdfReader.numPages
        alltext = []
        for i in range(numPages):
            page = pdfReader.getPage(i)
            text = page.extractText()
            alltext.append(text)
        corpus = []
        for i in alltext:
            review = re.sub('[^a-zA-Z]', ' ', i)
            if not i.strip(): continue
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)
        return open(path + "data/" + self.filename + ".txt", 'w+', encoding='utf-8').write('\n'.join(corpus))
        pdf.close()

    def tfidf(self):
        self.ReadPDFfile()
        text_files = glob.glob(path + "output/?.txt")
        text_files.insert(0, path + "data/" + self.filename + ".txt")
        documents = [open(f, encoding="utf8").read() for f in text_files]
        tfidf = TfidfVectorizer().fit_transform(documents)
        pairwise_similarity = (tfidf * tfidf.T).A
        x = pairwise_similarity[0][1:]
        print("hello", len(x))
        time = datetime.datetime.now()
        ranks = sorted([(j, i) for (i, j) in enumerate(x)], reverse=True)
        values = []
        posns = []
        for j, i in ranks:
            values.append(j)
            posns.append(i)
        if self.filenumber <= len(x):
            values = values[:self.filenumber]
            posns = posns[:self.filenumber]
            print(values, posns)

            for i in range(0, len(values)):
                print("The Percentage Of Plagiarism is = " + "{0:.0f}%".format(values[i] * 100))
                name = list(np.asarray(posns) + 1)
                print("fdhfhjtyu", name)
                namefile = text_files[name[i]]
                report.write("PLAGIARISM REPORT" + '\n')
                report.write('Project Name: ' + proname + '\n')
                report.write("The Percentage Of Plagiarism is = " + "{0:.0f}%".format(values[i] * 100) + '\n')
                report.write('File Name: ' + self.filename + '.pdf' + '\n' + 'Matched with: ' + namefile + '\n')
                report.write('Date: ' + time.strftime("%D") + '\n')

                myMatch = Matcher(path + "data/" + self.filename + '.txt', namefile, 2, 3)
                myMatch.match()
        else:
            report.write("PLAGIARISM REPORT" + '\n')
            report.write('Project Name: ' + proname + '\n')
            report.write('File Name: ' + self.filename + '.pdf' + '\n')
            report.write('Date: ' + time.strftime("%D") + '\n')
            report.write("The number of files out of range in the database" + '\n')

            print("The number of files out of range in the database" + '\n')
        Path(path + "data/" + self.filename + ".txt").replace(path + "output/" + self.filename + '.txt')


class Plagiarismtwofile:
    def __init__(self, filename1, filename):
        self.filename1 = filename1
        self.filename = filename

    def ReadPDFfile(self, file):
        pdf = open(path + "data/" + file, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdf)
        numPages = pdfReader.numPages
        alltext = []
        for i in range(numPages):
            page = pdfReader.getPage(i)
            text = page.extractText()
            alltext.append(text)
        corpus = []
        for i in alltext:
            review = re.sub('[^a-zA-Z]', ' ', i)
            if not i.strip(): continue
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)
        return open(path + "data/" + file + ".txt", 'w+', encoding='utf-8').write('\n'.join(corpus))
        pdf.close()

    def tfidf(self):
        self.ReadPDFfile(self.filename1)
        self.ReadPDFfile(self.filename)
        print("sddsad", self.filename1)
        print("sd", self.filename)
        text_files = [path + "data/" + self.filename1 + ".txt", path + "data/" + self.filename + ".txt"]
        documents = [open(f, encoding="utf8").read() for f in text_files]
        tfidf = TfidfVectorizer().fit_transform(documents)
        pairwise_similarity = (tfidf * tfidf.T).A
        x = pairwise_similarity[0][1:]
        print("hello", len(x), "     ", x)
        time = datetime.datetime.now()
        print("The Percentage Of Plagiarism is = " + "{0:.0f}%".format(max(x) * 100))
        report.write("PLAGIARISM REPORT" + '\n')
        report.write("The Percentage Of Plagiarism is = " + "{0:.0f}%".format(max(x) * 100) + '\n')
        report.write('File Name: ' + self.filename1 + '\n' + 'Matched with: ' + self.filename + '\n')
        report.write('Date: ' + time.strftime("%D") + '\n')

        myMatch = Matcher(path + "data/" + self.filename1 + '.txt', path + "data/" + self.filename + '.txt', 2, 3)
        myMatch.match()

        Path(path + "data/" + self.filename1 + ".txt").replace(path + "output/" + self.filename1 + '.txt')
        Path(path + "data/" + self.filename + ".txt").replace(path + "output/" + self.filename + '.txt')


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app.secret_key = 'many random bytes'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'graduation'
mysql = MySQL(app)


@app.route("/")
def index():
    cur = mysql.connection.cursor()
    print("jjjjjjjjjjj")
    cur.execute("SELECT  * FROM projects")
    data = cur.fetchall()
    cur.close()

    return render_template("filter-card-result.html", projects=data)


print("sssssssssssss")


@app.route('/result', methods=['POST', 'GET'])
def search_re():
    cur = mysql.connection.cursor()
    if request.method == 'POST':
        print("uuuuuuuuuuu")
        year = request.form['year']
        department = request.form['department']
        cur.execute("SELECT  * FROM projects where gyear='" + year + "'AND department='" + department + "'")
        data = cur.fetchall()
        cur.close()

        return render_template("filter-card-result.html", projects=data)


@app.route('/more', methods=['POST', 'GET'])
def document():
    cur = mysql.connection.cursor()
    curr = mysql.connection.cursor()
    if request.method == 'POST':
        global ip
        ip = request.form['more']
        cur.execute("SELECT  * FROM projects where pid='" + ip + "'")
        projects = cur.fetchall()
        for x in projects:
            global proname
            proname = x[0]
            doctor = x[3]
            department = x[5]
            description = x[7]
        cur.close()

        curr.execute("SELECT  * FROM project_team where projectid='" + ip + "'")
        teams = curr.fetchall()
        for x in teams:
            leadername = x[0]
            teamname = x[3]
            leaderemail = x[2]
        curr.close()
        print("ffffffffffff")
        return render_template("filter-card-more.html", description=description, leadername=leadername,
                               teamname=teamname, leaderemail=leaderemail, department=department,
                               ip=ip, proname=proname, doctor=doctor)


@app.route('/download')
def download_file():
    cur = mysql.connection.cursor()
    cur.execute("SELECT  * FROM projectdocument where pr_id='" + ip + "'")
    data = cur.fetchall()
    for x in data:
        pdfname = x[2]
        print(pdfname)
    cur.close()
    pdffile = path + "data/" + pdfname + ".pdf"
    print("done")
    return send_file(pdffile, as_attachment=True)


@app.route('/check', methods=['POST', 'GET'])
def report():
    cur = mysql.connection.cursor()
    curs = mysql.connection.cursor()
    if request.method == 'POST':
        cur.execute("SELECT  * FROM projectdocument where pr_id='" + ip + "'")
        data = cur.fetchall()
        for x in data:
            dname = x[2]
        cur.close()

        curs.execute("SELECT  document_name FROM projectdocument")
        name = curs.fetchall()
        cur.close()

        text_files = glob.glob(path + "output/?.txt")
        num = len(text_files)
        print("hjghgyyyyyyyyyy", num)
        global report, content
        report = open(path + "Report/" + dname + '.txt', 'w+', encoding='utf-8')
        number = int(request.form['quantity'])
        p = Plagiarism(number, dname)
        p.tfidf()
        report.close()
        fname = dname + ".txt"
        with open(path + "Report/" + fname, 'r') as f:
            cont = f.readlines()
        print(cont)
        content = ''.join(cont)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=15)
        f = open(path + "Report/" + fname, "r")
        for x in f:
            pdf.cell(200, 10, txt=x, ln=1, align='C')
        if os.path.exists(path + "Report/" + dname + ".pdf"):
            os.remove(path + "Report/" + dname + ".pdf")
        pdf.output(path + "Report/" + dname + ".pdf")
        global pdfpath
        pdfpath = path + "Report/" + dname + ".pdf"

        return render_template("report.html", content=content, num=num)


@app.route('/downloadreport')
def download_report():
    pdffile = pdfpath
    return send_file(pdffile, as_attachment=True)


@app.route('/check2', methods=['POST', 'GET'])
def goto():
    if request.method == 'POST':
        return render_template("check-2-files.html")


@app.route("/checktwo", methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'data/')
    if not os.path.isdir(target):
        os.mkdir(target)

    file1 = request.files['file1']
    file2 = request.files['file2']

    name1 = file1.filename
    destination = "/".join([target, name1])
    file1.save(destination)
    file1 = "\"%s\"" % file1
    file1 = file1.split("'")
    file1 = file1[1]

    name2 = file2.filename
    destination = "/".join([target, name2])
    file2.save(destination)
    file2 = "\"%s\"" % file2
    file2 = file2.split("'")
    file2 = file2[1]

    global report, content
    report = open(path + "Report/" + "file1" + '.txt', 'w+', encoding='utf-8')

    p = Plagiarismtwofile(file1, file2)
    p.tfidf()
    report.close()
    fname = "file1" + ".txt"
    with open(path + "Report/" + fname, 'r') as f:
        cont = f.readlines()
    print(cont)
    content = ''.join(cont)
    print(content)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    f = open(path + "Report/" + fname, "r")
    for x in f:
        pdf.cell(200, 10, txt=x, ln=1, align='C')
    if os.path.exists(path + "Report/" + "report.pdf"):
        os.remove(path + "Report/" + "report.pdf")
    pdf.output(path + "Report/" + "report.pdf")

    return render_template("report2file.html", content=content)


@app.route('/downloadreport2file')
def download_report2file():
    pdf2file = path + "Report/report.pdf"
    return send_file(pdf2file, as_attachment=True)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
