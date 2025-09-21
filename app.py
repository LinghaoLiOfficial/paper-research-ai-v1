import os
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_talisman import Talisman
from dotenv import load_dotenv, find_dotenv

from controller.login_controller import login_bp
from controller.file_controller import file_bp
from controller.web_crawl_controller import wc_bp
from controller.data_analysis_controller import da_bp
from controller.knowledge_base_controller import kb_bp
from controller.science_research_controller import sr_bp
from controller.visual_simulation_controller import vs_bp
from controller.test_controller import test_bp

# 加载全局环境变量
load_dotenv(find_dotenv(), verbose=True)

# 实例化Flask框架
app = Flask(__name__)

# 实例化CORS跨域协议
CORS(app, supports_credentials=True)

# 实例化Socketio框架
# socketio = SocketIO()
# socketio.init_app(app, cors_allowed_origins="*")

# 批量注册API蓝图
blueprints_list = [
    test_bp,
    login_bp,
    file_bp,
    wc_bp,
    da_bp,
    kb_bp,
    sr_bp,
    vs_bp
]
for blueprint in blueprints_list:
    app.register_blueprint(blueprint)


@app.get("/")
def default():
    return jsonify(200)


if __name__ == '__main__':
    app.run(
        host=os.getenv("HOST"),
        port=os.getenv("PORT"),
        debug=os.getenv("DEBUG_MODE"),
        # ssl_context=(os.getenv("SSL_PEM"), os.getenv("SSL_KEY")),
        threaded=True,
    )

