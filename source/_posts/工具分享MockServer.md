---
title: 工具分享MockServer
date: 2025-07-19 17:31:27
categories: "工具"
tags:
  - docker
  - mockserver
---
# [MockServer](https://github.com/mock-server/mockserver)

MockServer可用于建立模拟http/https请求响应的服务器，用于接口调试，可接受任意http请求，返回自定义内容。

[docker快速案例](https://www.mock-server.com/where/docker.html)

### 1. 安装docker
### 2. 拉取镜像 
```
   docker pull mockserver/mockserver
```
## 3. 配置文件准备
   需要准备一个mockeserver.properties和一个initializerJson.json文件：
```
   touch mockeserver.properties initializerJson.json
```
   mockeserver.properties具体内容参考[链接](https://www.mock-server.com/mock_server/configuration_properties.html), 这里只需要添加以下内容，并在initializerJson.json定义接口形式即可：
```
mockeserver.properties:
   mockserver.initializationJsonPath=/config/initializerJson.json

initializerJson.json:
[
    {
      "httpRequest": {
        "path": "/data/static/sync_data"      // 接口url定义
      },
      "httpResponse": {                       // 返回形式
        "body": {
            "code": 0,
            "errmsg": "OK"
        }
      }
    },
    {
      "httpRequest": {
        "path": "/status"                     // 定义多个url
      },
      "httpResponse": {
        "body": {
            "code": 0,
            "errmsg": "OK"
        }
      }
    }
]
```
## 4. 启动镜像
   定义完毕后，启动如下命令:
```
   docker run -d --rm --name mock -v $(pwd):/config -p 7535:7535  mockserver/mockserver -serverPort 7535
```
   --name用于定义启动镜像名称, 这里定义为mock方便启动和查看日志, -v表示映射某一文件夹到镜像某一位置，-p表示端口映射, mockserver/mockserver代表启动服务。
   启动完成后, 使用postman或其他形式调用接口，并可在日志中查看内容
```
   docker logs -f mock
```