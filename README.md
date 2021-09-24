# Tacotron Korean TTS implementation using pytorch

### Training

1. **한국어 음성 데이터 다운로드**

    * [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

2. **`~/Tacotron-pytorch`에 학습 데이터 준비**

   ```
   Tacotron-pytorch
     |- archive
         |- kss
             |- 1
             |- 2
             |- 3
             |- 4
         |- transcript.v.1.x.txt
   ```

3. **Preprocess**
   ```
   python preprocess.py
   ```
     * data 폴더에 학습에 필요한 파일들이 생성됩니다

4. **Train**
   ```
   python train1.py -n <name>
   python train2.py -n <name>
   ```
     * train1.py - train2.py 순으로 실행합니다
     * 원하는 name을 정하면 ckpt/<name> 폴더가 생성되고 모델이 저장됩니다
     * 재학습은 아래와 같이 로드할 모델 경로를 정해주면 됩니다
  
  ```
   python train1.py -n <name> -c ckpt/<name>/1/ckpt-<step>.pt
   python train2.py -n <name> -c ckpt/<name>/2/ckpt-<step>.pt
   ```
  
5. **Synthesize**
   ```
   python test1.py -c ckpt/<name>/1/ckpt-<step>.pt
   python test2.py -c ckpt/<name>/2/ckpt-<step>.pt
   ```
     * test1.py - test2.py 순으로 실행하면 output 폴더에 wav 파일이 생성됩니다



윈도우에서 Tacotron 한국어 TTS 학습하기
  * https://chldkato.tistory.com/141
  
Tacotron 정리
  * https://chldkato.tistory.com/143
