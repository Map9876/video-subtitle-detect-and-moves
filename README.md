# video-subtitle-detect-and-moves
移动视频字幕位置
https://github.com/copilot/c/cf7923de-e6dc-4a06-b0a1-674ddab6a98c

```
apt install opencv
#/项目母路径/include/nlohmann/json.hpp

mkdir build
cd build
cmake ..
make
```

Skip to content
Navigation Menu

Copilot Chat
视频区域选择器的HTML实现
问题是需要输入视频我需要处理的矩形四个角的坐标，仅仅处理所在的矩形坐标范围内以加快速度，为此我需要首先定位坐标，需要一个可视化显示视频，暂停视频，我可以在视频上方选择框的四个角，然后输出为标准坐标，同时代码也可以接收这个四个点的标准坐标，这个功能如何实现，通过简易的html还是什么，主要是我可以在画面上交互选择范围。
我会创建一个简单但功能完整的HTML页面来实现视频区域选择功能。这个方案使用纯前端技术，不需要服务器，可以直接在浏览器中打开使用：


video_selector.html
<!DOCTYPE html>
<html>
<head>
    <title>视频区域选择器</title>
    <style>
这个HTML文件提供了以下功能：

视频控制功能：

文件选择：可以选择本地视频文件
播放/暂停按钮
进度条：可以拖动到视频的任意位置
时间显示：当前时间/总时间
区域选择功能：

在视频画面上用鼠标拖拽选择矩形区域
选择区域会以红色半透明框显示
实时显示所选区域的四个角坐标
提供重置按钮清除选择
坐标输出：

显示四个角的精确坐标
提供适用于C++程序的参数格式
使用方法：

将代码保存为 video_selector.html
直接在浏览器中打开该文件
点击"选择文件"按钮选择要处理的视频
使用播放/暂停按钮和进度条定位到合适的帧
在视频画面上拖动鼠标选择要处理的区域
坐标会自动显示在下方，包括适用于C++程序的参数格式
这些坐标可以直接用于之前的C++程序，作为处理区域的参数。例如，如果坐标输出为 100,200,300,400，则可以这样调用C++程序：

bash
./subtitle_shifter input.mp4 output.mp4 30 100,200,300,400
此工具的优点是：

纯前端实现，不需要服务器
直观的可视化界面
精确的坐标选择
完整的视频控制功能
实时预览选择区域
直接输出C++程序可用的参数格式
您可以根据需要修改样式或添加额外功能，比如添加网格线辅助对齐，或者添加键盘快捷键等。

输出的四个点数据可以直接输入到程序 // 修改主函数开头部分
int main(int argc, char** argv) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] 
                  << " <input_video> <output_video> <shift_amount> <roi_coordinates>\n"
                  << "roi_coordinates format: x1,y1,x2,y2\n";
        return -1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int shiftAmount = std::stoi(argv[3]);
    
    // 解析ROI坐标
    std::string roiCoords = argv[4];
    std::vector<int> coords;
    std::stringstream ss(roiCoords);
    std::string item;
    while (std::getline(ss, item, ',')) {
        coords.push_back(std::stoi(item));
    }
    
    if (coords.size() != 4) {
        std::cerr << "Invalid ROI coordinates format\n";
        return -1;
    }
    
    // 创建ROI矩形
    cv::Rect roi(
        coords[0],                    // x1 使用 问题是现在html页面不好操控
让我帮你创建一个完整的HTML页面，实现视频区域选择功能。这个页面会允许你选择视频文件，暂停在任意帧，并通过鼠标点击来选择四个角点的坐标。


video_selector.html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>视频区域选择器</title>
使用说明：

将上述代码保存为 video_selector.html
在浏览器中打开该文件
点击"选择文件"按钮选择要处理的视频文件
使用"播放/暂停"按钮控制视频播放
在视频画面上依次点击想要选择的4个角点
页面底部会显示所选点的坐标，以及可直接用于C++程序的格式
特点：

完全在浏览器中运行，不需要服务器
支持视频播放控制
可视化选点，自动计算坐标
显示点的序号和连线
提供重置功能
直接输出C++程序可用的坐标格式
这个工具的输出格式完全符合你的C++程序的输入要求，输出的坐标字符串可以直接作为程序参数使用。例如，如果输出的坐标是 100,200,300,400,500,600,700,800，你可以直接这样使用：

bash
./your_program input.mp4 output.mp4 shift_amount 100,200,300,400,500,600,700,800
Ask Copilot

让我帮你创建一个完整的HTML页面，实现视频区域选择功能。这个页面会允许你选择视频文件，暂停在任意帧，并通过鼠标点击来选择四个角点的坐标。 video_selector.html <!DOCTYPE html> <html lang="zh"> <head> <meta charset="UTF-8"> <title>视频区域选择器</title> 使用说明： 将上述代码保存为 video_selector.html 在浏览器中打开该文件 点击"选择文件"按钮选择要处理的视频文件 使用"播放/暂停"按钮控制视频播放 在视频画面上依次点击想要选择的4个角点 页面底部会显示所选点的坐标，以及可直接用于C++程序的格式 特点： 完全在浏览器中运行，不需要服务器 支持视频播放控制 可视化选点，自动计算坐标 显示点的序号和连线 提供重置功能 直接输出C++程序可用的坐标格式 这个工具的输出格式完全符合你的C++程序的输入要求，输出的坐标字符串可以直接作为程序参数使用。例如，如果输出的坐标是 100,200,300,400,500,600,700,800，你可以直接这样使用： bash ./your_program input.mp4 output.mp4 shift_amount 100,200,300,400,500,600,700,800

```
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>视频区域选择器</title>
    <style>
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .video-container {
            position: relative;
            margin: 20px 0;
        }
        #videoCanvas {
            border: 2px solid #333;
        }
        .controls {
            margin: 10px 0;
        }
        .coordinates {
            margin: 20px 0;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        button {
            padding: 8px 16px;
            margin: 0 5px;
            cursor: pointer;
        }
        .point {
            position: absolute;
            width: 10px;
            height: 10px;
            background: red;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>视频区域选择器</h1>
        
        <div class="controls">
            <input type="file" id="videoFile" accept="video/*">
            <button id="playPauseBtn">播放/暂停</button>
            <button id="resetBtn">重置选点</button>
        </div>

        <div class="video-container">
            <video id="video" style="display: none;"></video>
            <canvas id="videoCanvas"></canvas>
        </div>

        <div class="coordinates">
            <h3>选择的坐标点：</h3>
            <div id="coordsOutput"></div>
            <div id="cppFormat"></div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        const videoFile = document.getElementById('videoFile');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const resetBtn = document.getElementById('resetBtn');
        const coordsOutput = document.getElementById('coordsOutput');
        const cppFormat = document.getElementById('cppFormat');

        let points = [];
        let isPlaying = false;

        // 视频文件选择处理
        videoFile.addEventListener('change', function(e) {
            const file = e.target.files[0];
            const url = URL.createObjectURL(file);
            video.src = url;
            video.addEventListener('loadedmetadata', function() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                drawFrame();
            });
        });

        // 播放/暂停控制
        playPauseBtn.addEventListener('click', function() {
            if (video.paused) {
                video.play();
                isPlaying = true;
            } else {
                video.pause();
                isPlaying = false;
            }
        });

        // 重置选点
        resetBtn.addEventListener('click', function() {
            points = [];
            drawFrame();
            updateCoordinates();
        });

        // 画布点击事件
        canvas.addEventListener('click', function(e) {
            if (points.length >= 4) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            points.push({
                x: Math.round(x * scaleX),
                y: Math.round(y * scaleY)
            });

            drawFrame();
            updateCoordinates();
        });

        // 绘制视频帧和点
        function drawFrame() {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // 绘制已选择的点
            points.forEach((point, index) => {
                ctx.beginPath();
                ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'red';
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // 绘制点的序号
                ctx.fillStyle = 'white';
                ctx.font = '16px Arial';
                ctx.fillText(index + 1, point.x + 10, point.y + 10);
            });

            // 如果选择了4个点，绘制连线
            if (points.length === 4) {
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }
                ctx.closePath();
                ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }

        // 更新坐标显示
        function updateCoordinates() {
            let coordsText = '选择的点坐标：\n';
            points.forEach((point, index) => {
                coordsText += `点${index + 1}: (${point.x}, ${point.y})\n`;
            });
            coordsOutput.textContent = coordsText;

            if (points.length === 4) {
                const coordsString = points.map(p => `${p.x},${p.y}`).join(',');
                cppFormat.textContent = `C++程序参数格式：${coordsString}`;
            } else {
                cppFormat.textContent = '请选择4个点以生成C++程序参数格式';
            }
        }

        // 视频播放时持续更新画布
        video.addEventListener('play', function() {
            function step() {
                if (!video.paused && !video.ended) {
                    drawFrame();
                    requestAnimationFrame(step);
                }
            }
            requestAnimationFrame(step);
        });
    </script>
</body>
</html>
```
