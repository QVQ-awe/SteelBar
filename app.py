# app.py
import cv2

from flask import Flask, request, jsonify, send_file
from infer import count_rebar
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/api/count/rebar', methods=['POST'])
def rebar_count_api():
    """钢筋计数API接口"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '图片文件为空'}), 400
    
    # 保存上传的图片
    filename = str(uuid.uuid4()) + '.jpg'
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    # 执行计数
    try:
        count, annotated_img = count_rebar(upload_path)
        # 保存结果图片
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(result_path, annotated_img)
        # 返回结果
        return jsonify({
            'count': count,
            'result_url': f'/api/result/{filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/result/<filename>')
def get_result(filename):
    """获取结果图片"""
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': '结果不存在'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)