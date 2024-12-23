import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import threading

# 创建主窗口
root = tk.Tk()
root.title("图像上色工具")
root.geometry("500x470")
root.minsize(500, 470)
root.configure(bg="#f4f4f4")

# 全局变量
image_paths = []
prompts = {}
output_dir = ""
progress_window = None
progress_label = None
preview_windows = {}  # 存储每个预览窗口的引用及其对应的重新生成按钮
processing_lock = threading.Lock()  # 锁定重新生成操作
image_size = (512, 512)  # 默认图像大小

# 初始化全局使用的模型和设置
device = "cuda" if torch.cuda.is_available() else "cpu"
controlnet = ControlNetModel.from_pretrained("ControlNet")
pipe = StableDiffusionControlNetPipeline.from_pretrained("Model",controlnet=controlnet)
pipe.to(device)

preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

negative_prompt = (
    "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4, "
    "text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, "
    "morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, "
    "blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, "
    "malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
)

# 美化按钮和标签的样式
button_style = {'bg': '#4CAF50', 'fg': 'white', 'font': ('Arial', 12, 'bold'), 'relief': 'raised', 'width': 20}
label_style = {'bg': '#f4f4f4', 'font': ('Arial', 12), 'width': 40, 'anchor': 'w'}


# 选择保存路径
def select_output_dir():
    global output_dir
    output_dir = filedialog.askdirectory(title="选择保存路径")
    if output_dir:
        output_label.config(text=f"保存路径: {output_dir}")
    else:
        output_label.config(text="未选择保存路径")


# 显示已选择的保存路径
output_label = tk.Label(root, text="未选择保存路径", **label_style)
output_label.pack(pady=5)


# 选择文件
def select_files():
    global image_paths
    image_paths = filedialog.askopenfilenames(title="选择要上色的图片", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    if image_paths:
        file_label.config(text=f"已选择文件: {len(image_paths)} 个文件")
    else:
        file_label.config(text="未选择文件")


# 显示已选择的文件数量
file_label = tk.Label(root, text="未选择文件", **label_style)
file_label.pack(pady=5)


# 获取提示
def get_prompts():
    for path in image_paths:
        show_prompt_dialog(path)


# 显示带有缩略图的提示输入对话框
def show_prompt_dialog(image_path):
    dialog = tk.Toplevel(root)
    dialog.title(f"输入提示 - {os.path.basename(image_path)}")

    img = Image.open(image_path)
    img.thumbnail((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(dialog, image=img_tk)
    img_label.image = img_tk
    img_label.pack(pady=10)

    prompt_entry = tk.Entry(dialog, width=50)
    prompt_entry.pack(pady=10)

    def submit():
        prompt = prompt_entry.get()
        prompts[image_path] = prompt if prompt else ""
        dialog.destroy()

    submit_button = tk.Button(dialog, text="提交", command=submit)
    submit_button.pack(pady=10)


# 预览图像函数
def preview_image(image, filename):
    preview_window = tk.Toplevel(root)
    preview_window.title(f"预览 - {filename}")

    img_tk = ImageTk.PhotoImage(image)

    img_label = tk.Label(preview_window, image=img_tk)
    img_label.image = img_tk
    img_label.pack(pady=10)

    def save_and_close():
        save_image(image, filename)
        preview_window.destroy()
        del preview_windows[filename]

    save_button = tk.Button(preview_window, text="保存并关闭", command=save_and_close)
    save_button.pack(pady=10)

    # 添加重新生成按钮
    regenerate_button = tk.Button(preview_window, text="重新生成",
                                  command=lambda: regenerate_image(filename, preview_window))
    regenerate_button.pack(pady=10)

    preview_windows[filename] = (preview_window, regenerate_button)


# 更新预览图像函数
def update_preview_image(colorized_image, filename):
    preview_image(colorized_image, filename)


# 重新生成图像函数
def regenerate_image(filename, preview_window):
    if processing_lock.locked():
        messagebox.showwarning("警告", "已有图像正在重新生成，请稍后再试。")
        return

    with processing_lock:
        disable_all_regenerate_buttons()
        image_path = [path for path in image_paths if os.path.basename(path) == filename][0]
        show_progress(filename)
        thread = threading.Thread(target=regenerate_image_threaded, args=(image_path, filename, preview_window))
        thread.start()


def regenerate_image_threaded(image_path, filename, preview_window):
    try:
        # 关闭当前预览窗口
        preview_window.destroy()

        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        prompt = prompts.get(image_path, "")

        with torch.no_grad():
            output = pipe(prompt=prompt, image=input_tensor, negative_prompt=negative_prompt,
                          num_inference_steps=50, guidance_scale=9.0)

        colorized_image = output.images[0]

        # 更新进度窗口
        update_progress()

        # 更新预览图像
        update_preview_image(colorized_image, filename)

    except Exception as e:
        messagebox.showinfo("错误", f"重新生成 {image_path} 时发生错误: {e}")

    progress_window.destroy()
    enable_all_regenerate_buttons()


# 保存图像到指定路径
def save_image(image, filename):
    timestamp = int(time.time())
    output_filename = f"colorized_image_{os.path.basename(filename)}_{timestamp}.{com.get()}"
    output_path = os.path.join(output_dir, output_filename)
    image.save(output_path)
    print(f"已保存: {output_path}")


# 显示进度窗口
def show_progress(filename):
    global progress_window, progress_label
    progress_window = tk.Toplevel(root)
    progress_window.title("处理进度")
    progress_window.geometry("300x100")
    progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
    progress_label = tk.Label(progress_window, text=f"正在处理图片: {filename}")
    progress_label.pack(pady=20)
    root.update_idletasks()


# 更新进度窗口
def update_progress():
    global progress_label
    progress_label.config(text="已完成处理")
    root.update_idletasks()


def on_size_change(event):
    selected_size = size_variable.get()
    width, height = map(int, selected_size.split('x'))
    global image_size, preprocess
    image_size = (width, height)
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    print(f"图像大小已更新为: {selected_size}")


# 开始处理图像
def process_images():
    if not image_paths:
        messagebox.showwarning("警告", "请先选择文件！")
        return

    if not output_dir:
        messagebox.showwarning("警告", "请先选择保存路径！")
        return

    show_progress(None)

    total_images = len(image_paths)
    processed_images = []

    for i, image_path in enumerate(image_paths):
        try:
            update_progress_label(i + 1, total_images, os.path.basename(image_path))

            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            prompt = prompts.get(image_path, "")

            with torch.no_grad():
                output = pipe(prompt=prompt, image=input_tensor, negative_prompt=negative_prompt,
                              num_inference_steps=50, guidance_scale=9.0)

            colorized_image = output.images[0]

            # 存储处理后的图像
            processed_images.append((colorized_image, os.path.basename(image_path)))

        except Exception as e:
            messagebox.showinfo("错误", f"处理 {image_path} 时发生错误: {e}")

    progress_window.destroy()
    messagebox.showinfo("完成", "图片处理完成")

    # 显示所有预览窗口
    for colorized_image, filename in processed_images:
        preview_image(colorized_image, filename)


# 使用线程处理图像
def start_processing():
    processing_thread = threading.Thread(target=process_images)
    processing_thread.start()


# 更新进度窗口标签
def update_progress_label(current, total, filename):
    global progress_label
    progress_label.config(text=f"正在处理图片 {current}/{total}: {filename}")
    root.update_idletasks()


# 禁用所有重新生成按钮
def disable_all_regenerate_buttons():
    for _, button in preview_windows.values():
        button.config(state=tk.DISABLED)


# 启用所有重新生成按钮
def enable_all_regenerate_buttons():
    for _, button in preview_windows.values():
        button.config(state=tk.NORMAL)


# 创建按钮
select_output_button = tk.Button(root, text="选择保存路径", command=select_output_dir, **button_style)
select_output_button.pack(pady=(20, 10))

select_button = tk.Button(root, text="选择文件", command=select_files, **button_style)
select_button.pack(pady=(10, 10))

get_prompts_button = tk.Button(root, text="输入提示", command=get_prompts, **button_style)
get_prompts_button.pack(pady=(10, 10))

process_button = tk.Button(root, text="开始处理", command=start_processing, **button_style)
process_button.pack(pady=(10, 10))

# 图像大小选择
size_label = tk.Label(root, text="请选择生成图像大小", **label_style)
size_label.pack(pady=5)
size_variable = tk.StringVar(value="512x512")
size_combobox = ttk.Combobox(root, textvariable=size_variable)
size_combobox.pack(pady=(5, 10))
size_combobox["value"] = ("512x512", "768x768", "1024x1024")
size_combobox.bind("<<ComboboxSelected>>", on_size_change)

# 文件格式选择
com_label = tk.Label(root, text="请选择生成图片格式", **label_style)
com_label.pack(pady=5)
xVariable = tk.StringVar()
com = ttk.Combobox(root, textvariable=xVariable)
com.pack(pady=(5, 10))
com["value"] = ("jpg", "jpeg", "png", "bmp")
com.current(0)

# 运行 Tkinter 主循环
root.mainloop()
































