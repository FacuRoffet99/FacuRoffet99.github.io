Title: Losses
Date: 2025-12-22 12:40
Category: Deep Learning
Lang: es
Slug: losses
Author: Facundo Roffet
Summary: Taxonomía estructurada de funciones de coste (loss functions) en deep learning, organizada por tipo de tarea y mecanismo objetivo. Cubre tareas discriminativas y generativas con énfasis en computer vision y machine learning moderno.

<!-- Hide default title -->
<style> h1.entry-title, h1.post-title, h1.title, h1:first-of-type {display: none;} </style>
<!-- Add custom title -->
<h2 style="text-align: center; font-size: 3em; color: rgba(12, 205, 76, 0.927);">Losses</h2>

<!---------------------------------------------------------------------------->

> Este post presenta una taxonomía estructurada de loss functions (funciones de coste) utilizadas en Deep Learning, organizándolas por tipo de tarea y mecanismo objetivo. La lista no es exhaustiva, ya que se centra en las losses más ampliamente adoptadas en la investigación moderna de computer vision y machine learning. Se omiten variantes de nicho o altamente especializadas, así como losses específicas para tareas sequence-to-sequence.

> La sección sobre tareas generativas sirve como una visión general amplia de términos en lugar de una lista granular de funciones independientes. Por el contrario, la sección discriminativa proporciona un desglose más detallado de formulaciones específicas.

> El objetivo de esta taxonomía es proporcionar una referencia intuitiva pero matemáticamente rigurosa para seleccionar una loss adecuada en función de los requisitos geométricos y probabilísticos de un problema específico. La categorización y las definiciones presentadas aquí se derivan principalmente de [Li et al. (2025)](https://doi.org/10.3390/math13152417).

<!---------------------------------------------------------------------------->

# this will be skipped

# Tareas discriminativas
Optimizan para $P(Y|X)$. Estas losses se centran en definir fronteras de decisión o ajustar funciones que mapean inputs directamente a targets.

## D1. Losses de regresión (continuas)
Los modelos de regresión apuntan a predecir una variable dependiente continua $y$ basada en variables independientes $x$. Las losses en esta categoría son funciones de los residuos: la diferencia entre el valor observado $y$ y el valor predicho $\hat{y} = f(x)$.

### D1a. Basadas en magnitud (punto a punto)
Estas losses miden el error punto a punto entre la predicción y el ground truth. Guían a los modelos para aproximar el valor objetivo minimizando la magnitud de estos errores.

**MAE (Mean Absolute Error)**  
$$ L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| $$
🗒️ Calcula el promedio de las diferencias absolutas entre los valores predichos y los reales.  
💡 "No me importa la dirección del error, solo decime en promedio por cuántas unidades le estoy errando. Además, no me voy a volver loco por outliers masivos."   
✅ Robusta a outliers (penalización lineal).  
✅ Proporciona una unidad física de error que es interpretable.  
❌ Los gradientes no son diferenciables en 0, lo que puede complicar la convergencia cerca del óptimo.  

**MSE (Mean Squared Error)**  
$$ L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$
🗒️ Calcula el promedio de las diferencias al cuadrado. Elevar el error al cuadrado asegura positividad y penaliza los errores grandes desproporcionadamente más que los pequeños.  
💡 "Los errores chicos están bien, pero si le errás por mucho te voy a castigar severamente para asegurar que no lo vuelvas a hacer."   
✅ Diferenciable en todas partes (descenso de gradiente suave).  
✅ Converge más rápido que MAE cuando está cerca del mínimo.  
❌ Altamente sensible a outliers, un solo punto de datos malo puede sesgar todo el modelo.  
↔️ Variante RMSE: Convierte el error nuevamente a las unidades originales de la variable objetivo tomando la raíz cuadrada, haciéndolo más fácil de interpretar.  
↔️ Variante RMSLE: Hace que la loss sea sensible a errores relativos en lugar de absolutos y penaliza la subestimación más que la sobreestimación.  

**Log-Cosh**  
$$ L_{LogCosh} = \frac{1}{N} \sum_{i=1}^{N} \log(\cosh(\hat{y}_i - y_i)) $$
🗒️ Calcula el logaritmo del coseno hiperbólico del error de predicción. Se aproxima a $\frac{x^2}{2}$ para $x$ pequeños y a $|x| - \log(2)$ para $x$ grandes.  
💡 "Actuá como MSE cuando el error sea chico para hacer un fine-tuning suave, pero pasate al comportamiento MAE cuando el error sea enorme así los outliers no te distraen."   
✅ Combina lo mejor de ambos mundos: robusta a outliers (como MAE) y diferenciable en todas partes (como MSE).  
❌ Computacionalmente más costosa que las losses polinómicas simples.  

**Huber**  
$$
L_{Huber} = 
\begin{cases}
  \frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \le \delta \\\\
  \delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{en otro caso}
\end{cases}
$$
🗒️ Una función a trozos que es cuadrática para errores pequeños (por debajo de un umbral $\delta$) y lineal para errores grandes. Requiere un hiperparámetro $\delta$ para definir el punto de transición.  
💡 "No entres en pánico si un punto de datos está muy lejos, simplemente traelo linealmente. Pero una vez que te acerques, curvá la loss para aterrizar el avión suavemente."   
✅ Robusta a outliers manteniendo la diferenciabilidad en 0.  
❌ Introduce un hiperparámetro ($\delta$) que debe ser ajustado.  

**Quantile**
$$
L_{Quantile} = 
\begin{cases}
  \tau |\hat{y}_i - y_i| & \text{si } |y - \hat{y}| \le \delta \\\\
  (1-\tau)|\hat{y}_i - y_i| & \text{en otro caso}
\end{cases}
$$ 
🗒️ Una extensión de MAE que aplica diferentes penalizaciones a la sobreestimación y subestimación basada en un cuantil elegido $\tau$ (entre 0 y 1). Se usa para predecir intervalos de predicción en lugar de una media única.  
💡 "No quiero solo el resultado promedio, quiero estar 90% seguro de que el valor real está por debajo de mi línea de predicción."   
✅ Permite la estimación de incertidumbre y la construcción de intervalos de confianza.  
❌ Más difícil de entrenar, la convergencia puede ser más lenta que con MSE/MAE estándar.  

### D1b. Geometry-aware (bounding boxes)
En tareas de detección de objetos, el objetivo de la regresión de bounding boxes es lograr una alineación geométrica entre la caja predicha y el ground truth. A diferencia de las losses basadas en magnitud, las losses geométricas no tratan las coordenadas de forma aislada. En su lugar ven la caja como una entidad geométrica unificada, optimizando la relación espacial (superposición, distancia y forma) entre la predicción y el ground truth.

**IoU (Intersection over Union)**  
$$ L_{IoU} = 1 - \frac{|B \cap B^{gt}|}{|B \cup B^{gt}|} $$
🗒️ Mide el área de superposición entre la caja predicha $B$ y la caja de ground truth $B^{gt}$ dividida por su área de unión.  
💡 "No me importa dónde están los píxeles exactamente, solo asegurate de que los dos cuadrados se superpongan lo máximo posible."   
✅ Invariante a la escala del problema (una caja chica y una grande con el mismo % de superposición tienen la misma loss).  
❌ Si las cajas no se superponen: IoU es 0, el gradiente es 0, y el modelo deja de aprender completamente.  
❌ Si las cajas se superponen completamente: IoU es 1, y el gradiente se vuelve 0 nuevamente.  

**GIoU (Generalized IoU)**  
$$ L_{GIoU} = 1 - IoU + \frac{|C \setminus (B \cup B^{gt})|}{|C|} $$
Donde $C$ es la caja convexa más pequeña que cubre tanto a $B$ como a $B^{gt}$.  
🗒️ Agrega un término de penalización basado en el espacio vacío dentro de la caja envolvente más pequeña $C$. Esto asegura que existan gradientes incluso cuando las cajas no se superponen.  
💡 "Si las cajas no se tocan, mové la predicción hacia el target para minimizar el espacio vacío entre ellas."   
✅ Resuelve el problema de desvanecimiento de gradiente para cajas que no se superponen.  
❌ No resuelve el problema de desvanecimiento de gradiente para cajas completamente superpuestas.  
❌ La convergencia es lenta, tiende a expandir la caja predicha para cubrir el target primero antes de encogerse para encajar.  

**DIoU (Distance IoU)**  
$$ L_{DIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} $$
Donde $\rho$ es la distancia Euclidiana, $b$ y $b^{gt}$ son los puntos centrales, y $c$ es la longitud diagonal de la caja envolvente.   
🗒️ Agrega una penalización que minimiza la distancia normalizada entre los puntos centrales de las dos cajas.  
💡 "No solo superpongas, apuntale al medio. Alineá los centros de las cajas directamente."   
✅ Converge mucho más rápido que GIoU porque minimiza la distancia directamente en lugar del área.  
✅ Resuelve completamente el problema de desvanecimiento de gradiente.  
❌ No considera la relación de aspecto de las cajas.  

**CIoU (Complete IoU)**  
$$L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
Donde $v$ mide la consistencia de la relación de aspecto y $\alpha$ es un parámetro de ponderación.  
🗒️ Extiende DIoU agregando un término para asegurar que la relación de aspecto de la predicción coincida con el target.  
💡 "Superponé, pegale al centro y asegurate de no estar dibujando un rectángulo alto cuando debería ser uno ancho."   
✅ Considera todos los factores geométricos: área de superposición, distancia del punto central y relación de aspecto.  
❌ El término de relación de aspecto $v$ es complejo y los gradientes a veces pueden ser inestables dependiendo de la implementación.  

**EIoU (Efficient IoU)**  
$$L_{EIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \frac{\rho^2(w, w^{gt})}{C_w^2} + \frac{\rho^2(h, h^{gt})}{C_h^2}$$
Donde $w,h$ son ancho/alto y $C_w, C_h$ son el ancho/alto de la caja envolvente.  
🗒️ Mejora CIoU dividiendo el término de relación de aspecto en penalizaciones separadas para las diferencias de ancho y alto.  
💡 "La matemática de CIoU es complicada, mejor vamos a medir el error de ancho y el error de alto por separado."   
✅ Convergencia más rápida y mejor precisión de localización que CIoU.  
✅ Resuelve la ambigüedad en CIoU donde diferentes pares $w/h$ podían producir la misma penalización de relación de aspecto.  

**SIoU (Scylla-IoU)**  
$$L_{SIoU} = 1 - IoU + \frac{\Delta + \Omega}{2}$$
Donde $\Delta$ es el costo de distancia y $\Omega$ es el costo de forma.  
🗒️ Introduce un costo angular a la regresión. Considera el ángulo vectorial entre el centro de la caja predicha y el ground truth. Prioriza alinear la caja al eje más cercano (X o Y) para minimizar la libertad de movimiento.  
💡 "Dejá de dar vueltas en diagonal. Movete estrictamente en horizontal o vertical para alinearte con el target primero, después ajustá el tamaño."   
✅ Converge más rápido que CIoU y EIoU reduciendo la oscilación de la caja durante el entrenamiento.  
❌ Computacionalmente un poco más pesada debido al cálculo de componentes trigonométricos (seno inverso).  

## D2. Losses de clasificación (discretas)
La clasificación es un subconjunto de tareas de aprendizaje supervisado donde el objetivo es asignar un input $x$ a una de $K$ clases discretas.

### D2a. Basadas en margen (fronteras de decisión)
Las losses de margen introducen un parámetro de umbral para imponer una separación mínima entre el puntaje predicho y la clase correcta. Obligan al modelo no solo a clasificar correctamente, sino a hacerlo con alta confianza manteniendo una 'distancia segura' de la frontera de decisión.

**Hinge**  
$$L_{Hinge} = \max(0, 1 - y_i \hat{y}_i)$$
🗒️ La loss estándar para Support Vector Machines (SVMs). Solo penaliza al modelo si el puntaje de la clase correcta no es suficientemente más alto que el margen. Si la predicción es correcta y segura ($y \hat{y} \ge 1$), la loss es cero.  
💡 "No quiero solo que tengas razón, quiero que tengas razón por un margen amplio. Si apenas pasás la línea de meta, igual te voy a penalizar."   
✅ Los puntos que se clasifican correctamente con alta confianza tienen gradientes 0 y no afectan la actualización del modelo, ahorrando recursos computacionales.  
❌ La función no es diferenciable en $y\hat{y}=1$, requiriendo métodos de optimización de sub-gradiente.  
↔️ Variante Squared Hinge: Diferenciable pero sensible a outliers.  
↔️ Variante Quadratic Smoothed Hinge: Lineal para errores grandes para mantener robustez, y cuadrática cerca de la frontera del margen para asegurar diferenciabilidad.  
↔️ Variante Ramp: Limita la loss para ignorar outliers extremos.  

**Exponential**  
$$L_{Exp} = e^{-y_i \hat{y}_i}$$
🗒️ Utilizada principalmente en algoritmos de boosting como AdaBoost. Aplica una penalización exponencial a márgenes negativos (clasificaciones incorrectas).  
💡 "Si te equivocás en un ejemplo difícil, la penalización va a ser masiva. Te tenés que obsesionar con los puntos de datos más difíciles."   
✅ Fuerza al modelo a enfocarse intensamente en los ejemplos en los que se está equivocando actualmente.  
✅ Diferenciable y convexa.  
❌ Debido a que la penalización crece exponencialmente, un solo outlier mal etiquetado puede dominar el gradiente y arruinar el proceso de entrenamiento.  

### D2b. Probabilísticas (divergencia de distribución)
Sea $q$ la distribución de probabilidad verdadera del dataset y $p_{\theta}$ la distribución predicha generada por el modelo. Las losses probabilísticas miden la divergencia (distancia) entre $q$ y $p_{\theta}$. Al minimizar esta divergencia, la distribución de salida del modelo converge hacia el ground truth.

**CE (Cross-Entropy)**  
$$L_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$
🗒️ Mide la diferencia de información entre la distribución predicha y la distribución verdadera. Cuando los targets están codificados en one-hot, minimizar CE es matemáticamente equivalente a maximizar la verosimilitud (likelihood) de la clase correcta.  
💡 "Si la imagen es un gato, quiero que la probabilidad de 'gato' sea 1.0. Cada pedacito de masa de probabilidad asignada a 'perro' o 'pájaro' aumenta la penalización."   
✅ La loss por defecto para clasificación, diferenciable y rigurosamente basada en Teoría de la Información.  
❌ Dominada por clases mayoritarias si los datos están desbalanceados.  
❌ Dominada por ejemplos fáciles (fondo) en tareas de detección densa.  
↔️ Variante Weighted CE: Multiplica la loss de la clase $k$ por un peso $\alpha_k$ (usualmente inverso a la frecuencia de clase o al número efectivo de muestras).  
↔️ Variante Label Smoothing: Cambia el target $y=1$ a $y=1-\epsilon$ y $y=0$ a $y=\frac{\epsilon}{K-1}$ para prevenir el overfitting.  

**Focal**  
$$L_{Focal} = -\frac{1}{N} \sum_{i=1}^{N} \alpha (1 - \hat{p}_i)^\gamma \log(\hat{p}_i)$$
🗒️ Agrega un factor modulador $(1 - \hat{p}_i)^\gamma$ a la Cross-Entropy estándar. Si una muestra ya está bien clasificada (ej., $\hat{p}_i = 0.9$), el factor se acerca a 0, silenciando efectivamente la loss para ese ejemplo.  
💡 "No me importa el cielo de fondo que ya identificaste correctamente 1.000 veces. Concentrate en ese único píxel difícil que parece un peatón."   
✅ Resuelve el problema de desbalance de clases sin sobremuestreo manual.  
✅ El estándar para detección de objetos densa.  
❌ Requiere ajustar dos hiperparámetros ($\alpha$ y $\gamma$) que pueden ser sensibles al dataset.  

**GHM (Gradient Harmonized Mechanism)**  
$$L_{GHM} = \sum_{i=1}^{N} \frac{L_{CE}}{GD(g_i)}$$
Donde $g_i$ es la norma del gradiente y $GD$ es la densidad de gradiente (una medida de cuántas muestras tienen ese mismo gradiente).  
🗒️ Una alternativa avanzada a Focal loss. Observa que los ejemplos fáciles y los outliers tienen comportamientos de gradiente distintos. Armoniza el entrenamiento normalizando la loss basada en la densidad de gradientes. Si un millón de ejemplos producen el mismo gradiente pequeño (fondo fácil), su contribución se divide por un factor de densidad grande.  
💡 "Si todos los demás están gritando lo mismo, voy a bajarle el volumen a ese grupo. Solo quiero escuchar los errores únicos/raros."   
✅ No requiere ajuste manual de $\alpha$ o $\gamma$ como Focal loss, se adapta a la dinámica de los datos de entrenamiento.  
❌ El costo computacional aumenta: requiere calcular un histograma de gradientes a través del batch/dataset durante el entrenamiento.  

**Poly**  

$$
L_{Poly} = -\sum_{j=1}^{\infty} \alpha_j (1 - \hat{p}_t)^j 
$$

🗒️ Trata a la Cross-Entropy como una expansión en serie de Taylor y la generaliza, permitiendo el ajuste de los coeficientes polinómicos principales $\alpha_j$ para cambiar estructuralmente cómo se comporta la loss (en lugar de tenerlos fijos en $1/j$). En la práctica, usualmente solo se modifica el coeficiente principal ($\epsilon_1$): $L_{Poly-1} = L_{CE} + \epsilon_1 (1 - \hat{p}_t)$.  
💡 "Cross-Entropy es una curva fija, pero podemos cambiarle la forma si lo necesitamos."   
✅ Función generalizada que abarca Cross-Entropy y Focal Loss como casos especiales.  
✅ Poly-1 suele superar a Focal/CE ajustando un solo parámetro, con un costo computacional adicional despreciable.  
❌ Introduce un hiperparámetro no estándar que debe encontrarse mediante grid search, ya que no hay un valor por defecto universal.  

### D2c. Overlap-based (segmentación)
En segmentación semántica, la clasificación ocurre a nivel de píxel. Sin embargo, las losses pixel-wise a menudo tienen problemas cuando el objeto objetivo ocupa solo una pequeña fracción de la imagen. Las losses basadas en superposición abordan esto optimizando directamente la intersección entre el mapa de segmentación predicho y el ground truth, priorizando la alineación global de la forma sobre la precisión individual de los píxeles.

**Tversky**  
$$L_{Tversky} = 1 - \frac{\sum \hat{y} y}{\sum \hat{y} y + \alpha \sum (1-y)\hat{y} + \beta \sum y(1-\hat{y})}$$
Donde $y$ es el target, $\hat{y}$ es la predicción, $\alpha$ controla la penalización para falsos positivos, y $\beta$ controla la penalización para falsos negativos.  
🗒️ Una generalización del coeficiente Dice (cuando $\alpha = \beta = 0.5$). Permite cambiar el balance entre precisión (evitar falsos positivos) y recall (evitar falsos negativos).  
💡 "Si encontrar el tumor es crítico y no podemos permitirnos perderlo, poné $\beta$ más alto que $\alpha$ para penalizar a los píxeles perdidos más que los que sobran."   
✅ Mucho mejor que Cross-Entropy para manejar desbalance en objetos pequeños.  
✅ Los parámetros $\alpha$ y $\beta$ dan flexibilidad para ajustar el trade-off en base a las necesidades clínicas o de negocio.  
❌ Puede ser inestable durante las etapas tempranas del entrenamiento comparada con CE pixel-wise.  
↔️ Variante Focal Tversky: Aplica el mecanismo de Focal loss al índice Tversky.  

**Sensitivity-Specificity**  
$$L_{SS} = w \cdot \frac{\sum (y-\hat{y})^2 y}{\sum y} + (1-w) \cdot \frac{\sum (y-\hat{y})^2 (1-y)}{\sum (1-y)}$$
🗒️ Optimiza explícitamente la suma ponderada de los errores cuadráticos para la clase positiva (sensibilidad) y la clase negativa (especificidad). Asegura que el modelo no logre alta precisión simplemente ignorando el fondo o el primer plano.  
💡 "Necesito que seas bueno encontrando el objeto, pero igual de bueno NO encontrando el objeto donde no existe. Balanceá tu entusiasmo."   
✅ Aborda tanto la sobre-segmentación (incluir demasiado fondo) como la sub-segmentación (perder partes del objeto).  
✅ Apropiada para contextos médicos donde la especificidad es tan vital como la sensibilidad.  
❌ Altamente sensible al parámetro de peso $w$. Si se configura incorrectamente, el modelo puede colapsar prediciendo solo el fondo o solo el primer plano.  

## D3. Losses métricas (espacio de embeddings)
El objetivo del aprendizaje métrico es aprender las distancias relativas entre inputs en lugar de predecir una etiqueta o valor específico. Las losses métricas operan sobre pares (o tripletes) de instancias de datos, extrayendo un embedding para cada una. Una métrica de distancia mide la similitud entre estas representaciones. El modelo se entrena para minimizar la distancia entre representaciones de inputs similares y maximizar la distancia entre los disímiles, estructurando el espacio de embeddings de manera significativa.

### D3a. Distancia Euclídea
Estas losses usan directamente la distancia geométrica en el espacio de embeddings como el objetivo de optimización.

**Contrastive Loss**  
$$L_{Contrastive} = \frac{1}{2} \sum_{i=1}^{N} [Y_iD_i^2 + (1-Y_i) \max(0, m - D_i)^2]$$
Donde $D = ||f(x_1) - f(x_2)||_2$ es la distancia Euclídea entre el par de muestras, $Y=1$ implica misma clase, $Y=0$ implica clase diferente, y $m$ es el margen.  
🗒️ Toma pares de muestras. Si pertenecen a la misma clase, minimiza su distancia. Si pertenecen a clases diferentes, las empuja hasta que estén al menos a un margen $m$ de distancia.  
💡 "Si son gemelos, abrácense. Si son desconocidos, aléjense hasta que tengan al menos 1 metro de espacio personal entre ustedes."   
✅ Es el enfoque fundacional simple para el aprendizaje métrico.  
❌ Difícil ajustar el margen. Si $m$ es muy chico, los clusters se superponen; si es muy grande, el entrenamiento se vuelve inestable.  

**Triplet**  
$$L_{Triplet} = \sum_{i=1}^{N} \max(0, D(a_i, p_i)^2 - D(a_i, n_i)^2 + m)$$
Donde $a$ es ancla, $p$ es positivo (misma clase), $n$ es negativo (clase diferente), y $m$ es margen.  
🗒️ Toma tres muestras a la vez: un ancla, un positivo y un negativo. Asegura que el ancla esté más cerca del positivo que del negativo por al menos un margen $m$.  
💡 "No me importa exactamente dónde está el ancla, siempre y cuando su amigo (positivo) esté más cerca de ella que su enemigo (negativo)."   
✅ Más flexible que Contrastive loss porque relaja la restricción sobre distancias absolutas, solo importa el ranking relativo.  
❌ Requiere encontrar negativos que estén actualmente más cerca que los positivos. Si se eligen negativos al azar, la loss suele ser 0 y el modelo no aprende nada.  

**InfoNCE (Information Noise-Contrastive Estimation)**  
🗒️ Trata la tarea como un problema de clasificación: "Entre este batch de $K$ negativos y 1 positivo, identificá el positivo". Maximiza la información mutua entre la query y la key positiva.  
💡 "Acá tenés una foto de un perro y 1.000 fotos de otras cosas. ¿Podés elegir el perro correcto de esta rueda de reconocimiento?"   
✅ Aprende de un positivo y muchos negativos simultáneamente, proporcionando una señal de gradiente mucho más rica que Triplet.  
✅ Es el backbone estándar para el aprendizaje de representación auto-supervisado moderno.  
❌ A menudo requiere un batch size muy grande para tener suficientes negativos difíciles y funcionar eficazmente.  

### D3b. Margen angular
Las losses basadas en márgenes angulares o coseno no optimizan directamente la posición absoluta y la distancia en el espacio de features. En cambio, se centran en las fronteras angulares entre clases proyectando features en una hiperesfera y optimizando la similitud coseno entre vectores de features y centros de clases.

**A-Softmax (Angular Softmax / SphereFace)**  
$$L_{Sphere} = -\log \frac{e^{||x_i|| \psi(\theta_{y_i})}}{e^{||x_i|| \psi(\theta_{y_i})} + \sum_{j \neq y_i} e^{||x_i|| \cos(\theta_j)}}$$
Donde $\psi(\theta)$ es una función monotónica que reemplaza $\cos(\theta)$ con $\cos(m\theta)$.  
🗒️ La primera loss angular. Introduce un margen angular multiplicativo $m$, y fuerza al ángulo de la clase correcta a ser $m$ veces más pequeño que el ángulo de cualquier clase incorrecta.  
💡 "Si el ángulo al centro de tu clase es 10 grados, voy a hacer de cuenta que en realidad es 40 grados ($m=4$). Tenés que trabajar 4 veces más duro para demostrar que pertenecés ahí."   
✅ Pionera en el concepto de márgenes angulares, demostrando que las restricciones geométricas en la hiperesfera mejoran significativamente la discriminación de features.  
❌ La optimización es difícil y requiere un annealing complejo del hiperparámetro $\lambda$ para converger.  

**AM-Softmax (Additive Margin Softmax / CosFace)**  
$$L_{Cos} = -\log (\frac{e^{s(\cos(\theta_{y_i}) - m)}}{e^{s(\cos(\theta_{y_i}) - m)} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}})$$
Donde $s$ es un factor de escala y $m$ es un margen de coseno aditivo.  
🗒️ Simplifica SphereFace moviendo el margen $m$ fuera de la función coseno. Resta un margen $m$ directamente del valor de similitud coseno.  
💡 "La Softmax estándar es muy permisiva. Le voy a restar 0.3 a tu puntaje de similitud. Efectivamente necesitás un puntaje de 1.3 para obtener un 1.0 perfecto. ¡Esforzate más!"   
✅ Mucho más fácil de implementar y entrenar que SphereFace.  
✅ Es más interpretable ya que optimiza directamente el gap de similitud coseno.  

**Additive Angular Margin (ArcFace)**  
$$L_{Arc} = -\log (\frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}})$$
Donde $m$ es un margen angular aditivo sumado dentro del coseno.  
🗒️ Agrega el margen $m$ dentro del término coseno, lo que corresponde a una penalización de distancia geodésica directa en la hiperesfera.  
💡 "Imaginate que las clases son países en un globo terráqueo. ArcFace dibuja fronteras estrictas con una zona buffer entre cada país directamente sobre la superficie de la esfera."   
✅ El margen tiene una correspondencia constante con la longitud de arco en la hiperesfera.  
✅ Es state-of-the-art para reconocimiento facial.  
❌ Requiere un ajuste cuidadoso de la escala $s$ y el margen $m$ dependiendo del ruido del dataset.  

**Quality Adaptive Margin Softmax (AdaFace)**  
$$L_{Ada} = -\log (\frac{e^{s \cos(\theta_{y_i} + g_{angle})}}{e^{s \cos(\theta_{y_i} + g_{angle})} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}})$$
Donde $g_{angle}$ es una función de margen que se adapta basada en la calidad de la imagen (norma del feature $||\hat{z}_i||$).  
🗒️ Adapta el margen basado en la calidad de la imagen de entrada. Aplica un margen estricto a imágenes de alta calidad y un margen relajado a imágenes de baja calidad para evitar que el modelo haga overfitting al ruido.  
💡 "Si la foto es HD, espero perfección. Si la foto es un cuadro borroso de una cámara de seguridad, voy a ser más suave con vos para que no te confundas tratando de aprender ruido."   
✅ Estado del arte para reconocimiento facial no restringido (ej., vigilancia, baja resolución).  
✅ Evita que el modelo se trabe tratando de optimizar muestras irreconociblemente.  
❌ Introduce complejidad en la implementación y depende de la asunción de que la norma del feature correlaciona con la calidad de la imagen (lo cual suele ser cierto, pero no siempre).  

<!---------------------------------------------------------------------------->

# Tareas generativas
Optimizan para $P(X)$ o $P(X,Y)$. Las losses para estas tareas se enfocan en aprender la distribución de datos subyacente para generar nuevas muestras o reconstruir inputs. Estas funciones objetivo suelen ser compuestas, mezclando múltiples términos (reconstrucción, coincidencia de distribución, calidad perceptual) para lograr resultados realistas.

## G1. Términos de reconstrucción (element-wise)
Estos términos aseguran fidelidad midiendo la diferencia directa entre el input original $x$ y el output reconstruido/generado $\hat{x}$.

**MSE (Mean Squared Error)**  
🗒️ Es la misma función usada para tareas discriminativas. Para tareas generativas, esto actúa como el término de "fidelidad".  
💡 "La imagen generada se tiene que ver exactamente como el input, píxel por píxel."   
✅ Simple de implementar y garantiza teóricamente el PSNR (Peak Signal-to-Noise Ratio) más alto.  
❌ En contextos generativos, MSE puro tiende a producir imágenes borrosas porque promedia los detalles de alta frecuencia.  

## G2. Términos de coincidencia de distribución (divergencias)
Estos términos minimizan la discrepancia estadística entre la distribución aprendida $P_g$ y la distribución de datos real $P_{data}$. Son centrales en GANs y Variational Autoencoders (VAEs).

**Minimax (GAN loss)**  
🗒️ Un juego de suma cero entre dos redes: un generador ($G$) intenta engañar al discriminador, y un discriminador ($D$) intenta distinguir real de falso.  
💡 "Generador: Apuesto a que te puedo engañar. Discriminador: No, no podés, voy a detectar el falso."   
✅ Produce detalles muy nítidos y realistas comparado con MSE.  
❌ Muy difícil de entrenar.  

**Wasserstein Distance (WGAN loss)**  
🗒️ Calcula el "trabajo" mínimo (masa × distancia) requerido para transformar una distribución en otra. A diferencia de la loss GAN estándar, el discriminador (ahora llamado crítico) emite un puntaje crudo, no una probabilidad.  
💡 "En lugar de preguntar '¿Verdadero o Falso?', mejor preguntá '¿Qué tan real es esto?' para dejarle saber al generador exactamente cuán lejos está del target, incluso si actualmente está fallando por completo."   
✅ Proporciona gradientes significativos incluso cuando las distribuciones real y falsa no se superponen en absoluto, resolviendo el problema de desvanecimiento de gradiente de las GANs estándar.  
✅ El valor de la loss correlaciona linealmente con la calidad visual de las imágenes generadas, lo cual no es cierto para la loss GAN estándar.  
❌ Requiere imponer continuidad 1-Lipschitz (el gradiente no puede cambiar muy rápido), lo cual es difícil de implementar.  

**KL (Kullback-Leibler Divergence)**  
$$ L_{KL} = \sum P(x) \log (\frac{P(x)}{Q(x)}) $$
🗒️ Mide cuánta información se pierde cuando la distribución $Q$ se usa para aproximar $P$. En VAEs, fuerza al espacio latente aprendido a seguir una distribución Gaussiana estándar.  
💡 "Mantené tu espacio latente organizado como una campana de Gauss estándar así podemos muestrear de él fácilmente después."   
✅ Fuerza a las variables latentes aprendidas a seguir una distribución tratable (usualmente Gaussiana Unitaria), asegurando que el espacio latente sea suave y continuo.  
❌ La restricción Gaussiana estricta a menudo resulta en salidas sobre-regularizadas y borrosas.  

**Sinkhorn Divergence**  
$$L_{Sinkhorn} = \min_{\pi} \sum_{i,j} C_{i,j} \pi_{i,j} + \epsilon H(\pi)$$
Donde $C$ es la matriz de costo, $\pi$ es el plan de transporte, y $H$ es la entropía de regularización.  
🗒️ Agrega un término de regularización entrópica al problema de transporte óptimo. Esto permite que la distancia Wasserstein sea calculada mucho más rápido usando el algoritmo Sinkhorn-Knopp.  
💡 "Calcular el plan perfecto de movimiento de tierra es difícil. Si permitimos un poco de aleatoriedad en a dónde va la tierra, podemos resolver la matemática 100 veces más rápido."   
✅ Diferenciable y computacionalmente lo suficientemente rápida para usarse como loss.  
❌ Si $\epsilon$ es muy grande, la métrica se vuelve demasiado borrosa y pierde la precisión geométrica de la distancia Wasserstein verdadera.  

## G3. Términos de difusión (eliminación de ruido)
Usados en Diffusion Probabilistic Models (DDPMs). El objetivo es revertir un proceso de ruido gradual.

**Simple Diffusion**   
🗒️ El modelo predice el ruido $\epsilon$ que fue agregado a la imagen $x_0$ en el paso de tiempo $t$.  
💡 "Te voy a mostrar una pantalla de TV con ruido. Decime exactamente qué píxeles son ruido para que pueda restarlos y revelar la imagen de abajo."   
✅ El entrenamiento es esencialmente un conjunto masivo de tareas de regresión (MSE sobre ruido), lo cual es mucho más estable comparado con GANs.  
❌ La inferencia es lenta, ya que generar una sola imagen requiere correr la red iterativamente (ej., 50 a 1000 veces) para eliminar el ruido paso a paso.  

**Denoising Score Matching**  
🗒️ Optimiza el modelo para estimar la función de score (el gradiente de la log-densidad de los datos). Al moverse a lo largo del gradiente, se mueve desde un punto de datos ruidosos hacia un punto de datos limpios.  
💡 "Te tiran en un bosque con niebla. No sabés dónde está la cima de la montaña, pero si mirás tus pies y pisás donde el suelo va hacia arriba, eventualmente vas a llegar."   
✅ Evita el problema intratable de calcular la constante de normalización de la distribución de probabilidad.  
❌ Técnicamente compleja de derivar e implementar comparada con el objetivo simplificado usado en Simple Diffusion.  

## G4. Términos de guía auxiliar (basados en features)
En lugar de comparar píxeles crudos, estas losses comparan representaciones de alto nivel extraídas por una red pre-entrenada.

**Perceptual**  
$$ L_{Perc} = || \phi(x) - \phi(\hat{x}) ||_2^2 $$
Donde $\phi$ es un extractor de features pre-entrenado.  
🗒️ Compara los mapas de activación internos de una red pre-entrenada para las imágenes reales y generadas.  
💡 "No me importa si el píxel exacto coincide. ¿La imagen parece un perro? ¿Los bordes y texturas coinciden con la percepción humana?"   
✅ Correlaciona mucho mejor con el juicio visual humano que MSE, y proporciona texturas excelentes para transferencia de estilo y super-resolución.  
❌ Depende de redes pre-entrenadas, por lo que puede fallar o producir artefactos si el dominio objetivo es vastamente diferente.  

**Style**  
$$ L_{Style} = || G(\phi(x)) - G(\phi(\hat{x})) ||_F^2 $$
Donde $G$ calcula la Matriz de Gram—correlaciones entre features.  
🗒️ Mide la correlación entre diferentes canales de features. Captura el "estilo" (textura, pinceladas, patrones de color) mientras descarta la estructura espacial.  
💡 "Capturá la onda de Van Gogh pero no te preocupes por dónde están ubicados los árboles."   
✅ Desacopla explícitamente la textura de la estructura, permitiendo la síntesis de patrones artísticos complejos sin necesitar datos de entrenamiento pareados.  
❌ No impone coherencia espacial, por lo que parches de textura pueden aparecer en ubicaciones semánticamente incorrectas (ej., pinceladas apareciendo en el cielo en lugar de los árboles).