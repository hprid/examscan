import csv
import html
import sys
import qrcode
from qrcode.image.base import BaseImage as QrBaseImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

class QrReportLab:
    def __init__(self, box_size, border, size):
        self.box_size = box_size
        self.border = border
        self.size = size
        self._matrix = []

    def add(self, row, col):
        self._matrix.append((row, col))

    def draw(self, canvas, x=0, y=0):
        for row, col in self._matrix:
            xr = (col + self.border) * self.box_size
            yr = (row + self.border) * self.box_size
            xr, yr = yr, -xr
            yr += self.size - self.box_size
            canvas.rect(xr + x, yr + y, self.box_size, self.box_size, stroke=0, fill=1)


class QrReportLabFactory(QrBaseImage):
    def drawrect(self, row, col):
        self._img.add(row, col)
    def save(self, stream, kind=None):
        raise NotImplementedError()

    def new_image(self, **kwargs):
        return QrReportLab(self.box_size, self.border, self.pixel_size)


class PDFGenerator:
    def __init__(self, outfile, exam):
        self._exam = exam
        self._style = ParagraphStyle(name='paragraph', fontName='Helvetica', fontSize=24,
                                     leading=28, alignment=TA_CENTER)
        self._doc = canvas.Canvas(outfile, pagesize=A4)

    def add_student(self, student_id, display_name):
        box_size = 4.5 * mm
        qr = qrcode.QRCode(version=4,
                           error_correction=qrcode.constants.ERROR_CORRECT_H,
                           box_size=box_size,
                           border=0,
                           image_factory=QrReportLabFactory)
        qr.add_data('s:%s:%s' % (student_id, self._exam))
        qr_image = qr.make_image()

        width, height = A4
        padding = 20 * mm

        code = qr_image.get_image()
        y_qr = height - padding - code.size
        x_qr = (width - code.size) / 2
        code.draw(self._doc, x_qr, y_qr)

        para = Paragraph('%s<br/>%s<br/><br/>%s' % (html.escape(display_name),
                                                   student_id,
                                                   self._exam),
                         style=self._style)
        para.wrap(width, height)
        para.drawOn(self._doc, 0, y_qr - 3 * padding)
        self._doc.showPage()

    def save(self):
        self._doc.save()


def main():
    with open('qr-codes.pdf', 'wb') as f_pdf:
        gen = PDFGenerator(f_pdf, sys.argv[1])
        with open(sys.argv[2]) as f:
            reader = csv.reader(f)
            for line in reader:
                student_id, last_name, first_name = line
                gen.add_student(student_id, '%s, %s' % (last_name, first_name))
        gen.save()

if __name__ == '__main__':
    main()
