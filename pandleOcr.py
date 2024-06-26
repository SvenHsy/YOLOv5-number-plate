from paddleocr import PaddleOCR,draw_ocr
# 显示结果
from PIL import Image

if __name__ == '__main__':

    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="japan")  # need to run only once to download and load model into memory
    img_path = './myImgs/001.jpg'
    result = ocr.ocr(img_path, cls=True)
    for line in result:
        print(line)



    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
