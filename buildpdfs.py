#!/usr/bin/env python3
import argparse
import base64
import glob
import hmac
import csv
import os
import re
import subprocess
import tempfile

from reportlab.lib.pagesizes import A5
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from pyzbar.pyzbar import decode as zbar_decode
from pyzbar.pyzbar import ZBarSymbol
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir', help='Destination directory')
    parser.add_argument('source_dir', help='Source directory (scanner output directory)')
    parser.add_argument('exam', help='Exam string (subdirectory of source directory)')
    parser.add_argument('secret', help='Secret for filename and password generation.')
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)
    exam_dir = os.path.join(args.source_dir, args.exam)
    student_ids = [student_id for student_id in os.listdir(exam_dir)
                   if student_id.isdigit()]
    with open(os.path.join(args.dest_dir, 'exams.csv'), 'w') as f:
        writer = csv.writer(f)
        for student_id in sorted(student_ids, key=int):
            image_files = glob.glob(os.path.join(exam_dir, student_id, '*.jpg'))
            image_files.sort()
            suffix_key = b'filename:' + args.secret.encode()
            suffix = hmac.new(suffix_key, student_id.encode(), 'sha256').digest()
            suffix = base64.b32encode(suffix).decode().lower().rstrip('=')
            filename = '%s_%s.pdf' % (student_id, suffix)
            output_file = os.path.join(args.dest_dir, filename)
            password_key = b'password:' + args.secret.encode()
            password = hmac.new(password_key, student_id.encode(), 'sha256').digest()
            password = base64.b64encode(password).decode().rstrip('=')
            password = password.replace('+', '').replace('/', '')
            with tempfile.NamedTemporaryFile('wb', suffix='.pdf') as f:
                render_pdf(f, image_files)
                subprocess.run(['qpdf', '--encrypt', password, password, '256',
                                '--', f.name, output_file], check=True)
            writer.writerow([student_id, filename, password])


def render_pdf(fileobj, image_files):
    doc = canvas.Canvas(fileobj, pagesize=A5)
    width, height = A5
    for i, pil_image in enumerate(iter_pil_images(image_files)):
        img_w, img_h = pil_image.size
        aspect = img_w / img_h
        img_width = width
        img_height = img_width / aspect
        if img_height > height:
            img_width *= height / img_height
            img_height = height
        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, '%d.jpg' % i)
            pil_image.save(filename, quality=80)
            doc.drawInlineImage(filename, 0, height - img_height, img_width,
                                img_height)
            doc.showPage()
    doc.save()
    fileobj.flush()


def iter_pil_images(image_files):
    for image_file in image_files:
        with Image.open(image_file) as img:
            width, height = img.size
            half_width = width // 2
            left = img.crop((0, 0, half_width, height))
            right = img.crop((half_width, 0, width, height))
            if not _is_empty_page(left):
                yield left
            if not _is_empty_page(right):
                yield right


def _is_empty_page(pil_image):
    qrcodes = zbar_decode(pil_image, symbols=[ZBarSymbol.QRCODE])
    for qrcode in qrcodes:
        qrtext = qrcode.data.decode().lower()
        if qrtext.startswith(('l:', 'r:')):
            return True
    return False


if __name__ == '__main__':
    main()
