# llava-image-captioning

LLaVA 멀티모달 기반 Image Captioning

## Installation

- `python==3.10` 환경에서 정상적으로 동작합니다.

- 아래 커맨드를 통해 패키지를 설치하실 수 있습니다.

```console
git clone https://github.com/leewaay/llava-image-captioning.git
cd llava-image-captioning

# Install LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .

cd ..
pip install -r requirements.txt
```

- DeepL API Key 설정

환경 변수를 사용하여 직접 설정:

```console
export DEEPL_KEY=YOUR_API_KEY_HERE
```

또는 `.env` 파일을 사용하여 설정:

1. 프로젝트의 루트 디렉토리에 `.env` 파일을 생성합니다.
2. `.env` 파일 내에 아래와 같이 API 키를 입력합니다.

```env
DEEPL_KEY=YOUR_API_KEY_HERE
```

이후 코드에서는 해당 키를 읽어 사용하게 됩니다.

<br>

## Usage

아래의 예시를 통해 서비스를 시작하고 종료할 수 있습니다.

- 서버 시작 (디폴트 포트 5010 사용):

```console
./image_captioning_server.sh start
```

- 서버 시작 (포트 8000 사용)

```console
./image_captioning_server.sh start 8000
```

- 서버 종료:

```console
./image_captioning_server.sh stop
```