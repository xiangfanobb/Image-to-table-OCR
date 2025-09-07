import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import pandas as pd
import os
import sys

def enhance_image_for_ocr(image_path):
    """
    专门针对OCR进行图像增强处理
    """
    try:
        # 使用PIL打开图像
        image = Image.open(image_path)
        
        # 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')
        
        # 增强对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # 增强锐度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # 二值化处理
        image = image.point(lambda x: 0 if x < 180 else 255, '1')
        
        # 轻微膨胀以连接断裂的字符
        image = image.filter(ImageFilter.MinFilter(3))
        
        return image
        
    except Exception as e:
        print(f"图像增强过程中出错: {e}")
        return None

def preprocess_image_with_opencv(image_path):
    """
    使用OpenCV进行高级图像预处理
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            # 如果OpenCV无法读取，尝试使用PIL
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

        denoised = cv2.medianBlur(gray, 3)
        

        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        

        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
        
    except Exception as e:
        print(f"OpenCV预处理过程中出错: {e}")
        return None

def extract_table_with_different_approaches(image_path):
    """
    尝试多种OCR方法和配置来提取表格
    """
    results = []
    
    # 方法1: 使用PIL增强后的图像
    enhanced_image = enhance_image_for_ocr(image_path)
    if enhanced_image:
        try:

            configs = [
                r'--oem 3 --psm 6',  
                r'--oem 3 --psm 4',  
                r'--oem 3 --psm 3',  
                r'--oem 3 --psm 11'  
            ]
            
            for config in configs:
                text = pytesseract.image_to_string(enhanced_image, config=config)
                if text.strip():
                    results.append(('PIL增强', config, text))
        except:
            pass
    
    # 方法2: 使用OpenCV预处理后的图像
    opencv_image = preprocess_image_with_opencv(image_path)
    if opencv_image is not None:
        try:
            pil_image = Image.fromarray(opencv_image)
            text = pytesseract.image_to_string(pil_image, config=r'--oem 3 --psm 6')
            if text.strip():
                results.append(('OpenCV预处理', 'psm6', text))
        except:
            pass
    
    # 方法3: 直接使用原始图像
    try:
        original_image = Image.open(image_path)
        text = pytesseract.image_to_string(original_image, config=r'--oem 3 --psm 6')
        if text.strip():
            results.append(('原始图像', 'psm6', text))
    except:
        pass
    
    return results

def extract_table_structure(image_path):
    """
    提取表格结构数据
    """
    best_result = None
    max_text_length = 0
    
    # 尝试多种方法
    results = extract_table_with_different_approaches(image_path)
    
    if not results:
        print("所有OCR方法都未能识别出文本")
        return None
    
    # 选择识别文本最多的结果
    for method, config, text in results:
        if len(text.strip()) > max_text_length:
            max_text_length = len(text.strip())
            best_result = text
            print(f"使用 {method} 方法 ({config}) 识别到 {len(text.strip())} 个字符")
    
    if best_result:
        # 按行分割文本
        lines = best_result.strip().split('\n')
        table_data = []
        
        for line in lines:
            if line.strip():  # 跳过空行
                # 尝试按制表符或空格分割
                if '\t' in line:
                    cells = line.split('\t')
                else:
                    # 尝试按多个空格分割
                    cells = [cell for cell in line.split('  ') if cell.strip()]
                    if len(cells) <= 1:
                        # 如果不能分割，整行作为一个单元格
                        cells = [line]
                
                table_data.append([cell.strip() for cell in cells if cell.strip()])
        
        return table_data
    
    return None

def visualize_ocr_results(image_path, results):
    """
    可视化不同方法的OCR结果
    """
    print("\n" + "="*60)
    print("OCR识别结果对比")
    print("="*60)
    
    for i, (method, config, text) in enumerate(results, 1):
        print(f"\n方法 {i}: {method} ({config})")
        print("-" * 40)
        if text.strip():
            lines = text.strip().split('\n')[:10]  # 只显示前10行
            for line in lines:
                print(f"  {line}")
            if len(text.strip().split('\n')) > 10:
                print("  ... (更多行)")
        else:
            print("  未识别到文本")
        print(f"  字符数: {len(text.strip())}")

def main():
    try:
        # 输入图像路径
        image_path = input("请输入图片路径: ").strip()
        
        # 移除可能的不可见字符
        image_path = image_path.replace('\u202a', '').replace('\u202c', '')
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 文件 '{image_path}' 不存在")
            return
        
        print("正在尝试多种OCR方法识别图片中的表格...")
        
        # 尝试多种方法并显示结果对比
        results = extract_table_with_different_approaches(image_path)
        visualize_ocr_results(image_path, results)
        
        # 提取最佳结果的表格结构
        table_data = extract_table_structure(image_path)
        
        if table_data is None or not any(table_data):
            print("\n未能成功识别表格内容")
            print("可能的原因:")
            print("1. 图像质量太差")
            print("2. 文字太小或模糊")
            print("3. 表格线干扰了识别")
            print("4. 需要更专业的OCR工具")
            return
        
        # 显示提取的表格数据
        print(f"\n提取到的表格数据 ({len(table_data)} 行):")
        print("-" * 50)
        for i, row in enumerate(table_data):
            print(f"行 {i+1}: {row}")
        
        # 询问是否保存为Excel
        save = input("\n是否保存为Excel文件? (y/n): ").strip().lower()
        if save == 'y':
            output_path = input("请输入输出Excel文件路径: ").strip()
            if not output_path.lower().endswith('.xlsx'):
                output_path += '.xlsx'
            
            # 保存为Excel
            try:
                df = pd.DataFrame(table_data)
                df.to_excel(output_path, index=False, header=False)
                print(f"表格已保存到: {output_path}")
            except Exception as e:
                print(f"保存Excel文件失败: {e}")
        
    except Exception as e:
        print(f"程序运行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置Tesseract路径
    try:
        pytesseract.get_tesseract_version()
        print("Tesseract OCR已找到在系统PATH中")
    except:
        # Windows安装路径
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        tesseract_found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Tesseract OCR已找到在: {path}")
                tesseract_found = True
                break
        
        if not tesseract_found:
            print("错误: 未找到Tesseract OCR引擎")
            sys.exit(1)
    
    main()
