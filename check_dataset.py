from datasets import load_dataset
import sys
sys.path.insert(0, 'c:\\Users\\simba\\Desktop\\Ethical')

print("Loading UC Berkeley Measuring Hate Speech dataset...")
ds = load_dataset('ucberkeley-dlab/measuring-hate-speech', split='train')

print(f'\n✅ Dataset is REAL!')
print(f'Total samples: {len(ds):,}')
print(f'\nColumns: {ds.column_names}')

print(f'\n📊 First 5 examples:')
for i in range(5):
    example = ds[i]
    text = example.get('text', 'N/A')
    hate_score = example.get('hate_speech_score', 'N/A')
    print(f'\n[{i+1}] Text: {text[:100]}...')
    print(f'    Hate Speech Score: {hate_score}')
    
print(f'\n📈 Score distribution analysis:')
scores = [ex.get('hate_speech_score', 0) for ex in ds.select(range(min(1000, len(ds))))]
import statistics
print(f'Mean: {statistics.mean(scores):.2f}')
print(f'Median: {statistics.median(scores):.2f}')
print(f'Min: {min(scores):.2f}')
print(f'Max: {max(scores):.2f}')

# Check label distribution with current mapping
labels = []
for score in scores:
    if score < 1.5:
        labels.append('NORMAL')
    elif score < 3.0:
        labels.append('OFFENSIVE')
    else:
        labels.append('HATE')

from collections import Counter
dist = Counter(labels)
print(f'\n🏷️ Label distribution (current mapping):')
for label, count in dist.items():
    print(f'{label}: {count} ({count/len(labels)*100:.1f}%)')