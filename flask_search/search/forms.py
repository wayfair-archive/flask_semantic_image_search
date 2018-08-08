from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import fields


class SearchForm(FlaskForm):

    image = FileField('Image', [FileRequired()])
    rows = fields.IntegerField('Rows', default=20)
    format = fields.RadioField('Format', choices=[('json', 'json'), ('html', 'html')])
