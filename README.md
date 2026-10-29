# tensight

- PyTorch の学習・推論の途中で、モデルの中のテンソル（値・データ型・分布・量子化の情報）を記録し、ブラウザで見るツール
- Python 3.14 以上と PyTorch 2.14 以上
- 量子化ライブラリ [quaver](https://github.com/Gucchi3/quaver) と一緒に使うのがおすすめ。quaver の量子化器を自動で見分け、scale やコードの使われ方、量子化の誤差も記録可能

## インストール

```bash
git clone https://github.com/Gucchi3/tensight.git
pip install -e ./tensight
```

## インポート
```python
import tensight as ts
```

## API

| 関数 | 説明 |
|---|---|
| `run_start(log_dir, name, kinds, interval, ...)` | 記録を始める。記録する種類（kinds）と間隔（interval。既定は 100 ステップ）などを決める |
| `watch_model(model, optimizer=None)` | モデルの層とパラメータに観測点をつける |
| `run_step()` | ステップを1つ進める。記録するステップなら書き込む |
| `run_close()` | 残りを書き込んで、記録を終える |
| `run_should_record()` | 次のステップを記録するかを返す（CUDA グラフや torch.compile で学習するとき） |
| `run_record_next()` | 次のステップを、間隔に関係なく記録する |
| `note_tensor(name, value)` | forward の途中の値に名前をつけて記録する |
| `inspect_tensor(value, before=None)` | 記録せずに、1つのテンソルの集計結果を返す |
| `adapter_register(adapter)` | ほかの量子化ライブラリの読み取り部品を足す |
| `converter_register(cls, converter)` | 独自の型の値からテンソルを取り出す関数を登録する |

## 使い方

```python
import tensight as ts

ts.run_start(log_dir="runs", name="int8_qat", kinds=("output", "weight", "grad"))
ts.watch_model(model, optimizer)
for x, y in loader:
    optimizer.zero_grad()
    criterion(model(x), y).backward()
    optimizer.step()
    ts.run_step()
ts.run_close()
```

記録フォルダ（この例では runs）の index.html をブラウザで開くと、記録を見られます。画面の見方は[使い方ガイド](docs/使い方ガイド.md)にあります。

## ライセンス

MIT。ただし、examples/cifar10/utils/quaver は quaver のコピーで、MPL-2.0 です。
