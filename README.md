# Review

## Overall
Не удалось обучить ни одну из моделей. Думаю, что в коде есть баг, который проявляется на example датасете. Вывод сделан после анализа предсказаний DeepSpeech2.

Подробности по финальным запускам **каждой** из моделей в example.ipynb

## Torch Deepspeech
Было замечено, что модель выдает nan predictions спустя несколько итераций.  
Попытки исправить:  
1. zero_infinity=True в CTCLoss
2. max_zero_grad=1 в конфиге.


не привели к результату

## Deepspeech2
Было принято решение протестировать deepspeech2.
1. Сходимость на onebatchtest наблюдается.
2. Сходимости на librispeech нет. Поведение метрик странное.

**Onebatchtest**
Все хорошо
<details>
<summary>Показать изображение</summary>

<img width="1151" height="324" alt="image" src="https://github.com/user-attachments/assets/96bbdcb3-cd7f-41ff-8cbd-e15d6da61d00" />
<img width="1156" height="322" alt="image" src="https://github.com/user-attachments/assets/d5adda1c-77b8-4408-a79f-14ad12c0788b" />
</details>

**Example**
Неадекватное поведение метрик. Нет падения лосса. В prediction выдает одну букву
<details>
<summary>Показать изображение</summary>
<img width="1156" height="329" alt="image" src="https://github.com/user-attachments/assets/59a7b34e-2858-442f-9076-0c467cab58e2" />
<img width="1158" height="310" alt="image" src="https://github.com/user-attachments/assets/fe5de1b0-7534-47e9-9496-b387e4bf54b5" />
<img width="1155" height="210" alt="image" src="https://github.com/user-attachments/assets/f7e0f0df-91e2-4e3a-a109-5b285800f061" />
</details>

## Conformer
Плохая сходимость в целом

**Onebatchtest**
<details>
<summary>Показать изображение</summary>
<img width="2421" height="630" alt="image" src="https://github.com/user-attachments/assets/f13395e4-042c-470b-8f5c-2e73ab9ebdee" />
<img width="2432" height="638" alt="image" src="https://github.com/user-attachments/assets/67b9bc4b-2b07-47fc-90b8-bdd2c241e992" />
</details>

**Example**
<details>
<summary>Показать изображение</summary>
<img width="1156" height="320" alt="image" src="https://github.com/user-attachments/assets/dee3ccd3-80a7-45c0-9fa4-ca7f3819d8cd" />
<img width="1156" height="335" alt="image" src="https://github.com/user-attachments/assets/daf70fc3-72a0-4ebf-8771-2b9dde842222" />
<img width="1156" height="280" alt="image" src="https://github.com/user-attachments/assets/3335c2f3-957d-4059-9e9c-c2b319838e15" />
</details>


