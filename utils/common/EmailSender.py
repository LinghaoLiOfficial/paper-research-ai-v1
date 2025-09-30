import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

from entity.common.StrucResult import StrucResult


class EmailSender:

    @classmethod
    def send_knowledge_graph_task_success(cls, email, task_id, task_name):

        html_content = f"""
            <div class="content">
                <p>您的以下任务以完成，请在个人界面查看任务结果~</p>
                <p>任务id: {task_id}</p>
                <p>任务名: {task_name}</p>
            </div>
        """

        return cls.send(
            email=email,
            html_title="论文知识网络应用程序通知",
            html_content=html_content
        )

    @classmethod
    def send_knowledge_graph_task_error(cls, email, task_id, task_name):

        html_content = f"""
            <div class="content">
                <p>您的以下任务运行出错，请检查并尝试手动重新开始任务</p>
                <p>任务id: {task_id}</p>
                <p>任务名: {task_name}</p>
            </div>
        """

        return cls.send(
            email=email,
            html_title="论文知识网络应用程序通知",
            html_content=html_content
        )

    @classmethod
    def send_knowledge_graph_task_cancel(cls, email, task_id, task_name):

        html_content = f"""
            <div class="content">
                <p>您的以下任务已被管理员强制终止，请手动重新开始任务</p>
                <p>任务id: {task_id}</p>
                <p>任务名: {task_name}</p>
            </div>
        """

        return cls.send(
            email=email,
            html_title="论文知识网络应用程序通知",
            html_content=html_content
        )

    @classmethod
    def send_validation_code(cls, email, validation_code):

        expiration_time = 10

        html_content = f"""
            <div class="content">
                <p>您的验证码如下: </p>
                <p class="code">{validation_code}</p>
                <p>该验证码仅在{expiration_time}分钟之内有效</p>
            </div>
        """

        return cls.send(
            email=email,
            html_title="注册验证码",
            html_content=html_content
        )

    @classmethod
    def send(cls, email, html_title, html_content):
        # 发件人和收件人信息
        sender_email = os.getenv("SMTP_SENDER_EMAIL")
        receiver_email = email

        sender_name = "LLH的个人网站"

        # 创建邮件
        message = MIMEMultipart()
        message["From"] = formataddr((sender_name, sender_email))
        message["To"] = receiver_email
        message["Subject"] = html_title

        # 添加HTML邮件内容
        email_html = f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{html_title}</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                width: 100vw;
                height: 100vh;
                display: flex;
                position: relative;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }}

            .content {{
                text-align: center;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }}

            .code {{
                font-size: 2em;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
        """

        message.attach(MIMEText(email_html, "html"))

        with smtplib.SMTP_SSL(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT"))) as server:
            server.login(sender_email, os.getenv("SMTP_CODE"))
            server.set_debuglevel(False)
            try:
                server.sendmail(sender_email, [receiver_email], msg=message.as_string())
                server.quit()
            except Exception as e:
                print(e)
                return StrucResult.build_error()

        return StrucResult.build_success()
