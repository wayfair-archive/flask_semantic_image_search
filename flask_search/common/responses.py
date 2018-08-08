from flask import jsonify


def invalid_request(error_message='Invalid Request'):
    response = jsonify({'error': error_message})
    response.status_code = 400
    return response


def server_error(error_message='Server Error'):
    response = jsonify({'error': error_message})
    response.status_code = 500
    return response


def page_not_found(e):
    response = jsonify({'error': 'Page Not Found'})
    response.status_code = 404
    return response


def healthy():
    response = jsonify({'status': 'UP'})
    response.status_code = 200
    return response
