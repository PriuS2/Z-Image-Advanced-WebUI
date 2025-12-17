# Z-Image Advanced WebUI

AI 이미지 생성을 위한 고급 WebUI 프로젝트입니다. Z-Image-Turbo-Fun-Controlnet-Union-2.0 모델을 기반으로 합니다.

## 주요 기능

- **노드 기반 이미지 생성**: React Flow를 활용한 직관적인 노드 에디터
- **Control 이미지 지원**: Canny, Depth, Pose, HED, MLSD 등 다양한 Control 타입
- **인페인팅**: 캔버스 기반 마스크 에디터로 부분 수정
- **LLM 프롬프트 향상**: OpenAI, Claude, Gemini, Ollama를 통한 번역 및 프롬프트 향상
- **갤러리 및 히스토리**: 생성 이미지 관리 및 프롬프트 히스토리
- **워크플로우 저장/불러오기**: 자주 사용하는 노드 구성 저장
- **배치 큐 시스템**: 다중 이미지 순차 생성
- **실시간 진행률 표시**: WebSocket 기반 실시간 업데이트
- **다국어 지원**: 한국어/영어

## 기술 스택

### Backend
- FastAPI + SQLAlchemy (SQLite)
- WebSocket 실시간 통신
- JWT 인증
- PyTorch + Diffusers

### Frontend
- React 18 + TypeScript
- React Flow (노드 에디터)
- Zustand (상태관리)
- Tailwind CSS + shadcn/ui
- react-i18next (다국어)

## 설치 방법

### 1. Python 가상환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
.\venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 2. Python 의존성 설치

```bash
# PyTorch with CUDA (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 기타 의존성
pip install -r requirements.txt
```

### 3. Frontend 의존성 설치

```bash
cd frontend
npm install
```

## 실행 방법

### Backend 서버 실행

```bash
# 프로젝트 루트에서
.\venv\Scripts\activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend 개발 서버 실행

```bash
cd frontend
npm run dev
```

## API 엔드포인트

### 인증 (`/api/auth`)
- `POST /register` - 회원가입
- `POST /login` - 로그인
- `GET /me` - 현재 사용자 정보

### 생성 (`/api/generation`)
- `POST /generate` - 이미지 생성 요청
- `POST /extract-control` - Control 이미지 추출
- `GET /status/{task_id}` - 작업 상태 조회
- `DELETE /cancel/{task_id}` - 작업 취소

### 갤러리 (`/api/gallery`)
- `GET /images` - 이미지 목록
- `DELETE /images/{id}` - 이미지 삭제
- `POST /images/{id}/favorite` - 즐겨찾기

### LLM (`/api/llm`)
- `POST /translate` - 한국어→영어 번역
- `POST /enhance` - 프롬프트 향상

### 워크플로우 (`/api/workflow`)
- `GET /` - 워크플로우 목록
- `POST /` - 워크플로우 생성
- `PUT /{id}` - 워크플로우 수정
- `DELETE /{id}` - 워크플로우 삭제

### WebSocket (`/ws/{user_id}`)
- 실시간 생성 진행률
- 단계별 미리보기
- 작업 취소

## 설정 파일

`config.yaml` 파일에서 다음 설정을 관리할 수 있습니다:

- 서버 설정 (호스트, 포트)
- JWT 설정
- 모델 경로
- GPU 최적화 옵션
- 생성 기본값
- LLM 제공자 설정

## 프로젝트 구조

```
Z-Image-Advanced-WebUI/
├── backend/
│   ├── main.py              # FastAPI 앱
│   ├── config.py            # 설정 관리
│   ├── api/                 # API 라우터
│   ├── services/            # 비즈니스 로직
│   ├── db/                  # 데이터베이스
│   └── websocket/           # WebSocket 핸들러
├── frontend/
│   ├── src/
│   │   ├── components/      # React 컴포넌트
│   │   ├── stores/          # Zustand 스토어
│   │   ├── api/             # API 클라이언트
│   │   ├── hooks/           # 커스텀 훅
│   │   └── i18n/            # 다국어 리소스
├── models/                  # AI 모델
├── outputs/                 # 생성 이미지
└── config.yaml              # 전역 설정
```

## 라이선스

MIT License
