# 『밑바닥부터 시작하는 딥러닝 ❷』

<a href="http://www.yes24.com/Product/Goods/72173703"><img src="https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/cover.png" width="150" align=right></a>

## 새소식
:white_check_mark: **2019.07.02** - 책 본문의 수식과 그림 파일들을 모아 공유합니다. 스터디 자료 등을 만드실 때 필요하면 활용하세요.

* [equations_and_figures_2.zip](https://github.com/WegraLee/deep-learning-from-scratch-2/blob/master/equations_and_figures_2.zip?raw=true)

---

## 시리즈 소개

<a href="https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/%EB%B0%91%EB%B0%94%EB%8B%A5%20%EC%8B%9C%EB%A6%AC%EC%A6%88%20%EC%86%8C%EA%B0%9C.pdf"><img src="https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/%EB%B0%91%EB%B0%94%EB%8B%A5%20%EC%8B%9C%EB%A6%AC%EC%A6%88%20%EC%86%8C%EA%B0%9C.png" width=1000></a>

『밑바닥부터 시작하는 딥러닝』 시리즈는 현재 4편까지 출간되었고, 2024년 중으로 5편도 출간될 예정입니다. 5편까지의 핵심 주제와 관계는 대략 다음 그림처럼 정리할 수 있습니다.

<img src="https://github.com/WegraLee/deep-learning-from-scratch-4/blob/master/series overview.png" width="600">

시리즈의 모든 책은 기존 편을 읽지 않았어도 무리가 없도록 꾸려졌습니다. 예를 들어 3편에서 만드는 프레임워크는 작동 원리뿐 아니라 API 형태까지 파이토치와 거의 같습니다. 그래서 3편을 읽지 않았어도 4편을 읽는 데 전혀 무리가 없습니다.

* [❶편의 깃허브 저장소](https://github.com/WegraLee/deep-learning-from-scratch)
* [❸편의 깃허브 저장소](https://github.com/WegraLee/deep-learning-from-scratch-3)
* [❹편의 깃허브 저장소](https://github.com/WegraLee/deep-learning-from-scratch-4)


## 동영상 강의
수원대학교 한경훈 교수님께서 『밑바닥부터 시작하는 딥러닝』 1, 2편을 교재로 진행하신 강의를 공개해주셨습니다. 책만으로 부족하셨던 분들께 많은 도움이 되길 바랍니다.

딥러닝 I - [강의 홈페이지](https://sites.google.com/site/kyunghoonhan/deep-learning-i)

[![ㅅㅣ리즈 1](https://img.youtube.com/vi/8Gpa_pdHrPE/0.jpg)](https://www.youtube.com/watch?v=8Gpa_pdHrPE&list=PLBiQZMT3oSxW1RS1hn2jWBgswh0nlcgQZ)

딥러닝 II - [강의 홈페이지](https://sites.google.com/site/kyunghoonhan/deep-learning-ii)

[![ㅅㅣ리즈 1](https://img.youtube.com/vi/5fwD1p9ymx8/0.jpg)](https://www.youtube.com/watch?v=5fwD1p9ymx8&list=PLBiQZMT3oSxXNGcmAwI7vzh2LzwcwJpxU)

딥러닝 III - [강의 홈페이지](https://sites.google.com/site/kyunghoonhan/deep-learning-iii)

[![ㅅㅣ리즈 1](https://img.youtube.com/vi/kIobK76on3s/0.jpg)](https://www.youtube.com/watch?v=kIobK76on3s&list=PLBiQZMT3oSxV3RxoFgNcUNV4R7AlvUMDx)

## 선수지식

다음은 역자가 추천하는 선수지식입니다.
<img src="https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/%EB%B0%91%EB%B0%94%EB%8B%A5%20%EC%84%A0%EC%88%98%EC%A7%80%EC%8B%9D.png" width=1000>

## 책 미리보기
[hanbit.co.kr](http://preview2.hanbit.co.kr/books/zcau/)

## 파일 구성

|폴더 이름 |설명                         |
|:--        |:--                          |
|ch01       |1장에서 사용하는 소스 코드 |
|ch02       |2장에서 사용하는 소스 코드    |
|...        |...                          |
|ch08       |8장에서 사용하는 소스 코드    |
|common     |공통으로 사용하는 소스 코드  |
|dataset    |데이터셋용 소스 코드 |

학습된 가중치 파일(6장, 7장에서 사용)은 아래 URL에서 받을 수 있습니다.
<https://www.oreilly.co.jp/pub/9784873118369/BetterRnnlm.pkl>

소스 코드에 관한 설명은 책을 참고하세요.

## 요구사항
소스 코드를 실행하려면 아래의 소프트웨어가 설치되어 있어야 합니다.

* 파이썬 3.x
* NumPy
* Matplotlib
 
또한 선택사항으로 다음 라이브러리를 사용합니다.

* SciPy
* CuPy


## 실행방법

각 장의 디렉터리로 이동한 후 python 명령을 실행하세요(**다른 디렉터리에서는 제대로 실행되지 않을 수 있습니다!**).

```
$ cd ch01
$ python train.py

$ cd ../ch05
$ python train_custom_loop.py
```

## 라이선스

이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다.
비상용뿐 아니라 상용으로도 자유롭게 이용하실 수 있습니다.


## 책의 오류

이 책의 오탈자 등 오류 정보는 [정오표](https://docs.google.com/document/d/1pzeh5nrP6y6A5WgT9vvxMpe-ai7ZRhU84BdAhdJzuFk/edit?usp=sharing)에서 확인하실 수 있습니다.
