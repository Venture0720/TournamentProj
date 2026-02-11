import wntr
import pandas as pd
import numpy as np

# 1. Создаем модель сети
wn = wntr.network.WaterNetworkModel()

# Добавляем узлы: Резервуар, Стык и Потребитель
wn.add_reservoir('res', base_head=20)
wn.add_junction('junc', base_demand=0.01, elevation=10)
wn.add_junction('leak_node', base_demand=0.02, elevation=10)

# Добавляем трубы
wn.add_pipe('pipe1', 'res', 'junc', length=100, diameter=0.3, roughness=100)
wn.add_pipe('pipe2', 'junc', 'leak_node', length=100, diameter=0.3, roughness=100)

# 2. Настраиваем время (моделируем 24 часа с шагом 15 минут)
wn.options.time.duration = 24*3600
wn.options.time.report_timestep = 900

# 3. Моделируем утечку
# В 12:00 дня на узле 'leak_node' появляется дырка (эмиттер)
leak_node = wn.get_node('leak_node')
wn.add_pattern('leak_pattern', [0]*48 + [1]*48) # Сначала нет утечки, потом есть
leak_node.add_leak(wn, area=0.01, start_time=12*3600)

# 4. Запуск гидравлического симулятора EPANET
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

# 5. Собираем данные в CSV
# Давление на узле и расход в трубе
pressure = results.node['pressure']['leak_node']
flow = results.link['flowrate']['pipe2']

df = pd.DataFrame({
    'Pressure (bar)': pressure * 0.1, # Перевод в бары
    'Flow Rate (L/s)': flow * 1000,   # Перевод в литры
    'Leak Status': [0 if t < 12*3600 else 1 for t in pressure.index]
})

df.to_csv('epanet_leak_data.csv', index=False)
print("Датасет на основе физики EPANET готов!")
