from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from math import sin,cos,pi
from random import Random
import numpy as np, shutil, zipfile, json

SRC=Path('/mnt/data/a_clean_stylized_3d_cgi_illustration_on_a_transpa.png')
OUT=Path('/mnt/data/TRAE宝_GIF表情包_6张')
OUT.mkdir(exist_ok=True)
S=480
FPS=14
N=21
DURATION=round(1000/FPS)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
LABELFONT=ImageFont.truetype(FONT, 53)
SMALLFONT=ImageFont.truetype(FONT, 19)
MINT=(109,255,199,255)
CYAN=(69,244,233,255)
WHITE=(255,255,255,255)
ORANGE=(255,167,87,255)
PINK=(255,91,132,255)
DARK=(11,19,25,255)

specs=[
 {'id':'01','slug':'收到','text':'收到！','kind':'salute','tile':(0,0),'hint':'敬礼 / ACK'},
 {'id':'02','slug':'让我想想','text':'让我想想','kind':'thinking','tile':(0,1),'hint':'思考 / LOADING'},
 {'id':'03','slug':'这也行','text':'这也行？','kind':'surprise','tile':(0,2),'hint':'惊讶 / WHAT'},
 {'id':'04','slug':'需求又改','text':'需求又改？','kind':'changes','tile':(1,0),'hint':'无奈 / UPDATE'},
 {'id':'05','slug':'起飞','text':'起飞！','kind':'rocket','tile':(1,1),'hint':'庆祝 / BOOST'},
 {'id':'06','slug':'辛苦啦','text':'辛苦啦','kind':'heart','tile':(1,2),'hint':'感谢 / LOVE'},
]

sheet=Image.open(SRC).convert('RGBA')
sprites=[]
for o in specs:
 r,c=o['tile']
 sub=sheet.crop((c*512,r*512,(c+1)*512,(r+1)*512))
 # a tiny bit of translucent glow at the edges is preserved
 bbox=sub.getchannel('A').point(lambda a: 255 if a>16 else 0).getbbox()
 cropped=sub.crop(bbox) if bbox else sub
 sprites.append(cropped)

# Shared palette for seamless looping colors; enough shades for the CGI gradient and pixel highlights.
sample=Image.new('RGB',(512*3,512*2),(5,8,13))
for i,spr in enumerate(sprites):
 x=(i%3)*512;y=(i//3)*512
 sm=spr.copy(); sm.thumbnail((470,470),Image.Resampling.LANCZOS)
 sample.paste(sm,(x+(512-sm.width)//2,y+(512-sm.height)//2),sm)
palette=sample.quantize(colors=255,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE)
pal=palette.getpalette()[:255*3] + [0,0,0]
# Pillow palette mode only supports 256 colors, reserve 255 for transparency
palsrc=Image.new('P',(1,1));palsrc.putpalette(pal)

# coordinate helpers

def circle(d,x,y,r,fill,outline=None,width=1):
 d.ellipse((int(x-r),int(y-r),int(x+r),int(y+r)),fill=fill,outline=outline,width=width)

def star(d,x,y,r,col,shape='diamond'):
 x=int(x);y=int(y);r=max(1,int(r))
 if shape=='diamond':
  d.polygon([(x,y-r),(x+r*.48,y-r*.40),(x+r,y),(x+r*.48,y+r*.40),(x,y+r),(x-r*.48,y+r*.40),(x-r,y),(x-r*.48,y-r*.40)],fill=col)
 else:
  d.line((x-r,y,x+r,y),fill=col,width=4); d.line((x,y-r,x,y+r),fill=col,width=4)

def heart(d,x,y,size,color):
 x=float(x);y=float(y);sz=float(size)
 p=[(x,y+sz*.9),(x-sz*.86,y+sz*.1),(x-sz*.84,y-sz*.34),(x-sz*.54,y-sz*.59),(x-sz*.20,y-sz*.55),(x,y-sz*.22),(x+sz*.20,y-sz*.55),(x+sz*.54,y-sz*.59),(x+sz*.84,y-sz*.34),(x+sz*.86,y+sz*.10)]
 d.polygon([(int(a),int(b)) for a,b in p],fill=color)

def bolt(d,x,y,scale,color):
 pts=[(-.32,-1),(.2,-1),(-.06,-.18),(.42,-.18),(-.35,1),(-.1,.1),(-.5,.1)]
 d.polygon([(int(x+a*scale),int(y+b*scale)) for a,b in pts],fill=color)

def waveform(d,phase,col):
 # coding status pulse on left / right
 for k in range(4):
  w=12+int(8*(1+sin(phase*2*pi+k)))
  x=18+k*17
  d.rounded_rectangle((x,380-w,x+7,380),radius=3,fill=col)

def effects_back(im,kind,t):
 d=ImageDraw.Draw(im,'RGBA')
 p=2*pi*t
 if kind=='salute':
  for i,(x,y) in enumerate([(75,115),(407,100),(416,276),(68,275)]):
   v=(sin(p+i*1.6)+1)*.5
   star(d,x,y,5+9*v,(97,255,190,int(120+100*v)))
  d.arc((37,84,440,397),10+int(10*sin(p)),172,fill=(63,245,190,65),width=3)
 elif kind=='thinking':
  # rhythmic orbiting loading symbols
  for i in range(10):
   a=2*pi*(i/10+t)
   x=240+200*cos(a); y=206+132*sin(a)
   circle(d,x,y,3+(i%3),(84,245,193,100+int(75*sin(a)**2)))
 elif kind=='surprise':
  for j,(x,y) in enumerate([(41,149),(416,75),(443,309),(71,308)]):
   sz=12+6*sin(2*pi*t*2+j)
   bolt(d,x,y,sz,(255,173,72,235))
 elif kind=='changes':
  for j in range(3):
   x=47+j*141+12*sin(p+j); y=110+(j%2)*60-19*sin(p+j*1.5)
   d.rounded_rectangle((x,y,x+73,y+52),radius=11,fill=(20,58,50,145),outline=(100,255,204,190),width=3)
   for k in range(3):
    d.rounded_rectangle((x+12,y+11+k*11,x+58-5*k,y+15+k*11),radius=2,fill=(117,255,210,230 if k<2 else 150))
 elif kind=='rocket':
  for j in range(9):
   rr=Random(j*21)
   x=60+rr.randint(0,365)
   y=285+rr.randint(0,90)+int((t*100+j*43)%88)
   star(d,x,y,3+(j%3)*2,(90,255,193,160))
  # visually pulsing launch rings
  rr=30+int(36*t)
  d.arc((240-rr*3,343-rr,240+rr*3,343+rr),0,180,fill=(57,252,180,int(95*(1-t)+15)),width=3)
 elif kind=='heart':
  for j in range(6):
   xx=50+j*70+7*sin(p+j)
   yy=290-((t*75+j*43)%215)
   heart(d,xx,yy,5+j%3,(255,85,124,170))


def effects_front(im,kind,t):
 d=ImageDraw.Draw(im,'RGBA');p=2*pi*t
 if kind=='salute':
  star(d,362,100,14+8*max(0,sin(p)),(155,255,209,230))
  if t<.60:
   d.rounded_rectangle((332,69,416,95),radius=9,fill=(27,61,51,195),outline=(111,255,202,205),width=2)
   d.text((374,80),'OK',anchor='mm',font=SMALLFONT,fill=WHITE)
 elif kind=='thinking':
  # dots appear in sequence on right upper side
  for i in range(3):
   alpha=215 if (int(t*9)%3)>=i else 60
   circle(d,320+i*27,51,8,(120,255,216,alpha))
 elif kind=='surprise':
  # flickering '!!!' and digital distortion blocks
  if sin(p*3)>.0:
   for i in range(3):
    x=337+i*22
    d.rounded_rectangle((x,29,x+8,55),radius=3,fill=(255,169,80,245))
    circle(d,x+4,64,4,(255,169,80,245))
 elif kind=='changes':
  # distinct incoming alert count balloons
  for i in range(2):
   xx=80+i*300; yy=83+8*sin(p+i)
   d.rounded_rectangle((xx-20,yy-18,xx+22,yy+18),radius=9,fill=(255,162,93,230))
   d.text((xx,yy),'!',font=ImageFont.truetype(FONT,27),anchor='mm',fill=DARK)
 elif kind=='rocket':
  # accelerating speed streaks
  for j in range(5):
   x=80+j*69+10*sin(p+j);y=346+j%2*12
   d.line((x,y,x,y+14+int(18*abs(sin(p+j)))),fill=(99,255,213,165),width=4)
 elif kind=='heart':
  for j in range(3):
   a=p+j*2*pi/3
   x=240+155*cos(a); y=170+85*sin(a)
   star(d,x,y,4+4*sin(a)**2,(255,189,211,200))

def draw_caption(im,sp,t):
 d=ImageDraw.Draw(im,'RGBA')
 txt=sp['text']
 font=LABELFONT
 bb=d.textbbox((0,0),txt,font=font,stroke_width=0)
 tw=bb[2]-bb[0]
 if tw>420:
  font=ImageFont.truetype(FONT,int(53*420/tw))
  bb=d.textbbox((0,0),txt,font=font);tw=bb[2]-bb[0]
 x=S//2
 # transparent soft footer to maintain readability on all chat themes
 d.rounded_rectangle((max(22,int((S-tw)/2)-22),398,min(S-22,int((S+tw)/2)+22),473),radius=22,fill=(12,23,29,184))
 d.rounded_rectangle((max(22,int((S-tw)/2)-22),398,min(S-22,int((S+tw)/2)+22),473),radius=22,outline=(65,229,176,130),width=2)
 # bold white text with chunky graphite keyline and mint glow shadow
 y=426+1.5*sin(2*pi*t)
 d.text((x+2,y+3),txt,font=font,anchor='mm',fill=(46,230,164,205),stroke_width=5,stroke_fill=(13,37,33,180))
 d.text((x,y),txt,font=font,anchor='mm',fill=(255,255,255,255),stroke_width=3,stroke_fill=(6,14,18,255))


def render(sp,spr,i):
 t=i/N
 frame=Image.new('RGBA',(S,S),(0,0,0,0))
 effects_back(frame,sp['kind'],t)
 # central chibi figure, varying motion per expression
 p=2*pi*t
 if sp['kind']=='salute':
  angle=1.7*sin(p); bob=-7*abs(sin(p)); scale=1+.023*sin(p)
 elif sp['kind']=='thinking':
  angle=2.6*sin(p);bob=-5*sin(p);scale=1+.017*sin(p)
 elif sp['kind']=='surprise':
  angle=3.8*sin(2*p);bob=3*sin(2*p);scale=1.02+.03*abs(sin(p*2))
 elif sp['kind']=='changes':
  angle=-1.8+2.5*sin(p);bob=4*abs(sin(p));scale=1+.014*sin(p)
 elif sp['kind']=='rocket':
  angle=2.3*sin(p);bob=-20*(.5+.5*sin(p));scale=1+.025*sin(p)
 else:
  angle=2.4*sin(p);bob=-6*sin(p);scale=1+.030*sin(p)
 # choose cropped alpha-visible sprite fit ~ 400x355
 maxw,maxh=430,355
 f=min(maxw/spr.width,maxh/spr.height)*scale
 w,h=round(spr.width*f),round(spr.height*f)
 s=spr.resize((w,h),Image.Resampling.LANCZOS)
 s=s.rotate(angle,Image.Resampling.BICUBIC,expand=True)
 x=int(S/2-s.width/2)
 y=int(195-s.height/2+bob)
 frame.alpha_composite(s,(x,y))
 effects_front(frame,sp['kind'],t)
 draw_caption(frame,sp,t)
 return frame


def palettize(rgba):
 # transparent pixels are encoded at reserved 255; preserve true opaque colors
 rgb=rgba.convert('RGB')
 p=rgb.quantize(palette=palsrc,dither=Image.Dither.NONE)
 arr=np.array(p)
 aa=np.array(rgba.getchannel('A'))
 arr[aa<115]=255
 dst=Image.fromarray(arr,'P');dst.putpalette(pal)
 dst.info['transparency']=255
 dst.info['disposal']=2
 return dst

frames_preview=[]
for sp,spr in zip(specs,sprites):
 rgbaframes=[render(sp,spr,i) for i in range(N)]
 png=OUT/f"{sp['id']}_{sp['slug']}.png"
 rgbaframes[5].save(png,optimize=True)
 palframes=[palettize(fr) for fr in rgbaframes]
 name=OUT/f"{sp['id']}_{sp['slug']}.gif"
 palframes[0].save(name,save_all=True,append_images=palframes[1:],duration=[DURATION]*N,loop=0,disposal=2,transparency=255,optimize=False)
 print(f'{name.name} {name.stat().st_size/1048576:.2f} MiB, {len(palframes)} frames')
 frames_preview.append(rgbaframes[5])

# contact sheet preview; background is for presentation only, GIFs themselves are transparent
W,H=1272,910
preview=Image.new('RGB',(W,H),(14,21,29))
d=ImageDraw.Draw(preview)
fontbig=ImageFont.truetype(FONT,36)
fontminor=ImageFont.truetype(FONT,19)
d.text((40,26),'TRAE宝  /  六连发表情包',font=fontbig,fill=(244,255,250))
d.text((42,83),'DEVELOPER REACTIONS  •  6 LOOPING GIF STICKERS',font=fontminor,fill=(130,216,187))
for ix,(sp,fr) in enumerate(zip(specs,frames_preview)):
 col=ix%3; row=ix//3
 xx=38+col*414; yy=125+row*385
 d.rounded_rectangle((xx,yy,xx+384,yy+360),radius=28,fill=(27,37,46),outline=(65,90,91),width=2)
 sm=fr.resize((360,360),Image.Resampling.LANCZOS)
 preview.paste(sm,(xx+12,yy),sm)
 d.text((xx+16,yy+327),sp['id']+'   '+sp['hint'],font=fontminor,fill=(183,228,213))
preview_path=OUT/'预览_六宫格.png'
preview.save(preview_path,optimize=True)

# bundled editable/reference assets and provenance
shutil.copy2(SRC,OUT/'角色原始设定_六姿态.png')
readme=f'''# TRAE宝 | 开发者日常嘴替 | 6 GIF 表情包

生成日期：2026-10-26
作品主题：让 TRAE宝成为开发者和日常聊天里的情绪嘴替。

## 内容
- 01 收到！：敬礼 + 绿光 OK 状态闪烁
- 02 让我想想：思考漂浮 + 省略号逐个点亮
- 03 这也行？：惊讶晃动 + 电子叹号闪烁
- 04 需求又改？：无奈摊手 + 新需求面板与感叹号飘入
- 05 起飞！：喷射上升 + 速度线与像素粒子
- 06 辛苦啦：抱心摆动 + 爱心粒子漂浮

## 输出规格
- 480×480 像素，1:1 比例
- 每张约 {N} 帧，循环时长约 {N*DURATION/1000:.2f} 秒，透明背景 GIF
- 含每张 PNG 静态样张、六宫格预览、角色姿态原图与 Python 制作脚本

## 官方参赛要求和重要说明
官方要求使用 TRAE 完成主要创作流程，并保留截图作为证明。
这一版本的初始角色图由 ChatGPT 图像生成功能生成；GIF 动画由本地 Python/Pillow 合成。
**它不是经由 TRAE 创作的参赛完成稿，不能把本制作过程冒充为 TRAE 完成。**
要正式投稿，应把角色设定、参考图与本脚本导入 TRAE，由 TRAE 执行主要设计/动画复刻、优化和导出；请实际保存完整开发聊天、文件修改记录和作品效果截图。
正式投稿时还应保留你自己的创意说明，并按官方要求发布至「案例与作品」。

官方活动链接：https://forum.trae.cn/t/topic/462975

## 可交给 TRAE 的制作提示词
请以本项目的 `角色原始设定_六姿态.png` 作为视觉参考，严格保留 TRAE宝黑色方块屏幕头、荧光绿色像素眼睛、黑色机身、精致轻 3D/Q 版科技风。阅读并改进 `create_gifs.py`，用 TRAE 完成实质性的动画重设计、逐帧效果开发、文案微调和 GIF 导出。输出 6 张统一系列的 480×480 透明 GIF：收到！（敬礼及 OK 闪烁）、让我想想（漂浮及加载动画）、这也行？（惊讶晃动及叹号）、需求又改？（无奈与新需求弹窗）、起飞！（喷射与速度线）、辛苦啦（抱心及心形粒子）。要求每帧角色外形一致、文字清晰、循环流畅、GIF 文件经过体积优化；另导出对照预览图、列出主要修改及真实创作过程截图。
'''
(OUT/'创作说明_及TRAE二创提示词.md').write_text(readme,encoding='utf-8')

ZIP=Path('/mnt/data/TRAE宝_6张GIF表情包.zip')
with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(OUT.iterdir()):
  if f.is_file():z.write(f,arcname=f.name)
print('zip',ZIP.stat().st_size/1048576, 'MiB')
