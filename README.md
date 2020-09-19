<h2>ANN4EEG</h2>

В файле func.py описываются все сети. В main.py происходит обучение. load_nn.py загружает уже обученную сеть и проверяет её работу на тестовых датасетах, которые формируются по тем правилам.

Скрипт dataset_generator.py создает датасет из файлов .txt, которые содержат отсчеты сигналов и предварительно созданы в EDFbrowser. Желательно использовать при формировании датасета для каждого препарата по 5 файлов сигнала длительностью по несколько минут снятых с разных особей. При этом название .txt файлов имеет вид "qwe_nn_data.txt", где qwe - первые три буквы названия препарата (например, для амитриптилина это "ami"). Все файлы должны лежать в одной директории с dataset_generator.py. Длительности выборки для одного сигнала опреляется параметром sig_len в dataset_generator.py. Скрипт сам разобьет сигналы на обучающее и тестовое множество.
Кроме того, скрипт присваивает сигналам метки классов в формате "qwe". Комментарии на руском с описанием методов и параметров есть во всех скриптах.

<h2>Dataset</h2>

Download i-EEG dataset_set_1000.zip (274.7 Мб): http://dx.doi.org/10.17632/gmkbhj28jh.1

<h2>Implementation</h2>

https://cmi.to/r2/

<h2>Authors</h2>

1. Alexey A. Nevzorov, Institute of Mathematics and Information Technology, Volgograd State University, Volgograd, Russian Federation
2. Konstantin Y. Kalitin, Department of Pharmacology and Bioinformatics, Volgograd State Medical University, Volgograd, Russian Federation
