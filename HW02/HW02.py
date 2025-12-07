import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

figures_dir = 'C:/Users/79854/OneDrive/Рабочий стол/HW02/figures'
os.makedirs(figures_dir, exist_ok=True)

np.random.seed(42)
n_rows = 1000

data = {
    'customer_id': np.arange(1, n_rows + 1),
    'age': np.random.normal(35, 10, n_rows).astype(int),
    'gender': np.random.choice(['М', 'Ж'], n_rows, p=[0.48, 0.52]),
    'income': np.random.normal(50000, 15000, n_rows),
    'purchase_amount': np.random.exponential(100, n_rows),
    'purchase_count': np.random.poisson(3, n_rows),
    'category': np.random.choice(['Электроника', 'Одежда', 'Продукты', 'Книги', 'Косметика'], n_rows,
                                 p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'region': np.random.choice(['Москва', 'СПб', 'Новосибирск', 'Екатеринбург', 'Казань'], n_rows,
                               p=[0.3, 0.2, 0.2, 0.15, 0.15]),
    'is_premium': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
    'satisfaction_score': np.random.randint(1, 11, n_rows),
    'days_since_last_purchase': np.random.exponential(30, n_rows).astype(int)
}

df = pd.DataFrame(data)

df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
df.loc[np.random.choice(df.index, 30), 'age'] = np.nan
df.loc[np.random.choice(df.index, 20), 'purchase_amount'] = np.nan

df.loc[10, 'age'] = 150
df.loc[20, 'income'] = -5000
df.loc[30:35, 'purchase_amount'] = -100

duplicate_row = df.iloc[0].copy()
duplicate_row['customer_id'] = 1001
df = pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=True)

print("Первые 5 строк датасета:")
print(df.head())

print("Информация о датасете:")
df.info()

print("Базовые описательные статистики:")
print(df.describe())

print("Статистики для категориальных переменных:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}:")
    print(df[col].value_counts().head())

missing_data = df.isna().sum()
missing_percentage = (df.isna().mean() * 100).round(2)
missing_df = pd.DataFrame({
    'Пропущено': missing_data,
    'Процент': missing_percentage
})
print("Анализ пропущенных значений:")
print(missing_df[missing_df['Пропущено'] > 0])

plt.figure(figsize=(10, 6))
missing_df[missing_df['Пропущено'] > 0]['Процент'].plot(kind='bar', color='coral')
plt.title('Процент пропусков по столбцам')
plt.xlabel('Столбцы')
plt.ylabel('Процент пропусков (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{figures_dir}/missing_values.png', dpi=300)
plt.show()

full_duplicates = df.duplicated().sum()
print(f"Полностью дублирующих строк: {full_duplicates}")
if 'customer_id' in df.columns:
    customer_duplicates = df['customer_id'].duplicated().sum()
    print(f"Дубликатов customer_id: {customer_duplicates}")

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(f"{col}: {negative_count} отрицательных значений")

if 'age' in df.columns:
    unrealistic_age = df[(df['age'] > 100) | (df['age'] < 0)].shape[0]
    if unrealistic_age > 0:
        print(f"age: {unrealistic_age} нереалистичных значений")

if 'income' in df.columns:
    zero_income_purchase = df[(df['income'] == 0) & (df['purchase_count'] > 0)].shape[0]
    if zero_income_purchase > 0:
        print(f"{zero_income_purchase} записей: доход 0, но есть покупки")

print("Выводы по качеству данных:")
print(f"""
Обнаруженные проблемы:
1. Пропуски в income (5%), age (3%), purchase_amount (2%)
2. Дубликаты: {full_duplicates} полных дубликатов
3. Подозрительные значения: отрицательные суммы покупок, нереалистичный возраст
4. Логические несоответствия: нулевой доход при наличии покупок
Рекомендации: заполнить пропуски медианами, удалить дубликаты, исправить аномалии.
""")

if 'gender' in df.columns:
    gender_dist = df['gender'].value_counts()
    print("Распределение по полу:")
    print(gender_dist)

if 'region' in df.columns:
    region_dist = df['region'].value_counts()
    print("Распределение по регионам:")
    print(region_dist)

if 'region' in df.columns and 'income' in df.columns:
    region_stats = df.groupby('region').agg({
        'income': ['mean', 'median', 'count'],
        'purchase_amount': 'mean',
        'purchase_count': 'sum'
    }).round(2)
    print("Статистики по регионам:")
    print(region_stats)

if 'age' in df.columns:
    bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    if 'income' in df.columns:
        age_stats = df.groupby('age_group', observed=True).agg({
            'income': 'mean',
            'purchase_amount': 'mean',
            'customer_id': 'count'
        }).round(2)
        print("Анализ по возрастным группам:")
        print(age_stats)

if 'income' in df.columns:
    plt.figure(figsize=(12, 8))
    income_clean = df['income'].dropna()
    n_bins = 30

    plt.hist(income_clean, bins=n_bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)

    mean_income = income_clean.mean()
    median_income = income_clean.median()

    plt.axvline(mean_income, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_income:.0f}')
    plt.axvline(median_income, color='green', linestyle='--', linewidth=2, label=f'Медиана: {median_income:.0f}')

    plt.title('Распределение дохода клиентов')
    plt.xlabel('Доход (руб.)')
    plt.ylabel('Плотность вероятности')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{figures_dir}/income_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

if 'income' in df.columns and 'region' in df.columns:
    plt.figure(figsize=(12, 8))

    data_to_plot = []
    regions = sorted(df['region'].unique())

    for region in regions:
        region_data = df[df['region'] == region]['income'].dropna()
        if len(region_data) > 0:
            data_to_plot.append(region_data)

    box = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=regions)

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold', 'violet']
    for patch, color in zip(box['boxes'], colors[:len(regions)]):
        patch.set_facecolor(color)

    plt.title('Распределение дохода по регионам')
    plt.xlabel('Регион')
    plt.ylabel('Доход (руб.)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{figures_dir}/income_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

if all(col in df.columns for col in ['income', 'purchase_amount', 'is_premium']):
    plt.figure(figsize=(12, 8))

    scatter_data = df[['income', 'purchase_amount', 'is_premium']].dropna()

    regular = scatter_data[scatter_data['is_premium'] == 0]
    premium = scatter_data[scatter_data['is_premium'] == 1]

    plt.scatter(regular['income'], regular['purchase_amount'], alpha=0.6, label='Обычные', color='blue', s=50)
    plt.scatter(premium['income'], premium['purchase_amount'], alpha=0.6, label='Премиум', color='red', s=50)

    z_reg = np.polyfit(regular['income'], regular['purchase_amount'], 1)
    p_reg = np.poly1d(z_reg)
    plt.plot(regular['income'], p_reg(regular['income']), "b--", alpha=0.8, linewidth=2)

    z_prem = np.polyfit(premium['income'], premium['purchase_amount'], 1)
    p_prem = np.poly1d(z_prem)
    plt.plot(premium['income'], p_prem(premium['income']), "r--", alpha=0.8, linewidth=2)

    plt.title('Зависимость суммы покупки от дохода')
    plt.xlabel('Доход (руб.)')
    plt.ylabel('Сумма покупки (руб.)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    corr_reg = np.corrcoef(regular['income'], regular['purchase_amount'])[0, 1]
    corr_prem = np.corrcoef(premium['income'], premium['purchase_amount'])[0, 1]
    plt.text(0.05, 0.95, f'Корреляция (обычные): {corr_reg:.2f}', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.text(0.05, 0.90, f'Корреляция (премиум): {corr_prem:.2f}', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

    plt.savefig(f'{figures_dir}/income_vs_purchase_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

if 'category' in df.columns and 'purchase_amount' in df.columns:
    plt.figure(figsize=(12, 8))

    category_stats = df.groupby('category')['purchase_amount'].mean().sort_values(ascending=False)

    bars = plt.bar(category_stats.index, category_stats.values, color='lightseagreen', edgecolor='black', alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.0f}', ha='center', va='bottom', fontsize=12)

    plt.title('Средняя сумма покупки по категориям товаров')
    plt.xlabel('Категория товара')
    plt.ylabel('Средняя сумма покупки (руб.)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    plt.savefig(f'{figures_dir}/purchase_by_category.png', dpi=300, bbox_inches='tight')
    plt.show()

saved_files = []
if os.path.exists(figures_dir):
    for file in os.listdir(figures_dir):
        if file.endswith('.png'):
            saved_files.append(file)

if saved_files:
    print(f"Всего сохранено графиков: {len(saved_files)}")