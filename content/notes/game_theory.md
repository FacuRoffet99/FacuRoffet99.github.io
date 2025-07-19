Title: Teoría de juegos
Date: 2025-07-19 20:10
Category: Toma de decisiones
Lang: es
Slug: decision_making_1
Author: Facundo Roffet

<!-- Hide default title -->
<style> h1.entry-title, h1.post-title, h1.title, h1:first-of-type {display: none;} </style>
<!-- Add custom title -->
<h2 style="text-align: center; font-size: 3em; color: rgba(12, 205, 76, 0.927);">Teoría de juegos</h2>

<!---------------------------------------------------------------------------->

> Estas son mis notas personales del curso [Game Theory with Ben Polak](https://www.youtube.com/playlist?list=PL6EF60E1027E1A10B) de Yale, y del video [What Game Theory Reveals About Conflict and War](https://www.youtube.com/watch?v=mScpHTIi-kM&pp=ygUWdmVyaXRhc2l1bSBnYW1lIHRoZW9yeQ%3D%3D) de Veritasium. 

<!---------------------------------------------------------------------------->

## Definiciones iniciales

*   Un juego está compuesto necesariamente por jugadores, sets de estrategias y recompensas. Por ejemplo, los jugadores pueden ser 1 y 2, los sets de estrategias ser S₁={T,B} y S₂={L,C,R} y las recompensas ser u₁(T,C)=11 y u₂(T,C)=3.
*   En una instancia particular de un juego, cada jugador elige una estrategia de su set y se determina un perfil de estrategias "s" para ese juego. Por ejemplo, s₁=T y s₂=C conforman el vector s=(T,C).
*   Una estrategia sᵢ' domina estrictamente a otra estrategia propia sᵢ si la recompensa para sᵢ' es estrictamente mayor que la de sᵢ sin importar lo que hagan los demás jugadores: uᵢ(sᵢ', s₋ᵢ) > uᵢ(sᵢ, s₋ᵢ) para todo s₋ᵢ.
*   En el caso de que la recompensa de sᵢ' sea mayor o igual a la de sᵢ (para cualquier estrategia de los demás jugadores), entonces sᵢ' domina débilmente a sᵢ: uᵢ(sᵢ', s₋ᵢ) ≥ uᵢ(sᵢ, s₋ᵢ) para todo s₋ᵢ.

## Lecciones del dilema del prisionero

*   Cooperar es una estrategia estrictamente dominada: los jugadores racionales no van a elegirla.
*   La elección racional puede derivar en resultados malos para todos los jugadores: la racionalidad individual no siempre conduce al bien colectivo.
*   Cambiar las recompensas puede alterar considerablemente al juego: no se puede obtener lo que uno quiere hasta no saber qué es lo que uno quiere.
*   Si los demás jugadores poseen estrategias estrictamente dominantes, hay que jugar en consecuencia de ellas: ponerse en los zapatos de los demás para determinar qué es lo que van a hacer.

## Eliminación iterativa de estrategias dominadas

*   Como ningún jugador racional elegirá una estrategia estrictamente dominada, las mismas pueden ser eliminadas. De esta forma, se forma un "juego reducido" en el que nuevamente se puede repetir el proceso hasta que no puedan eliminarse más estrategias. 
*   Esto solo aplica en el caso de que exista conocimiento común de racionalidad: vos creés que los demás jugadores van a actuar en forma racional, ellos creen que vos vas actuar en forma racional, vos creés que ellos creen que vas a actuar en forma racional, etc.

## Mejores respuestas

*   Una estrategia sᵢ' es una mejor respuesta ante la estrategia s₋ᵢ de los demás jugadores si la recompensa de elegir sᵢ' es mayor o igual que elegir cualquier otra estrategia: uᵢ(sᵢ',s₋ᵢ) ≥ uᵢ(sᵢ,s₋ᵢ) para todo sᵢ. Por lo tanto, sᵢ' maximiza uᵢ(sᵢ,s₋ᵢ) con respecto a sᵢ.
*   Si un juego no se puede resolver eliminando estrategias dominadas (porque no hay o porque el set de estrategias es continuo), entonces hay que tener en cuenta las creencias propias (en porcentaje) de lo que los demás jugadores van a hacer.
*   Una estrategia sᵢ' es una mejor respuesta ante la creencia p sobre las decisiones de los demás jugadores si la esperanza de recompensa al elegir sᵢ' es mayor o igual que la esperanza de elegir cualquier otra estrategia: E[uᵢ(sᵢ',p)] ≥ E[uᵢ(sᵢ,p)] para todo sᵢ. Por lo tanto, sᵢ' maximiza E[uᵢ(sᵢ,p)] con respecto a sᵢ.
*   Una estrategia es racionalizable si es compatible con al menos una de las posibles creencias sobre los demás jugadores. Esto significa que no hay que elegir estrategias que no sean mejores respuestas ante ninguno de los casos posibles.

## Equilibrio de Nash (NE)

*   Un perfil de estrategias s' es NE si la estrategia elegida por cada jugador (sᵢ') es una mejor respuesta ante las estrategias elegidas por cada otro jugador (s₋ᵢ').
*   Un NE implica que no hay arrepentimiento en los jugadores: manteniendo las estrategias ajenas fijas, nadie tiene incentivos estrictos para desviarse de su estrategia. También se puede decir que es una profecía autocumplida: creer que los demás van a jugar un NE hace que uno también lo haga.
*   Teorema de Existencia de Nash: todo juego finito (en jugadores y estrategias) tiene al menos un NE si se permiten estrategias mixtas. 
*   Una estrategia estrictamente dominada jamás puede ser jugada en un NE.
*   Un juego tiende a converger a un NE (en caso de que exista) tras ser repetido varias veces.
*   Sí un jugador se desvía de un NE pero los demás jugadores prefieren mantener sus estrategias, entonces el NE es robusto ante pequeñas variaciones. En cambio, si a los demás jugadores también les conviene cambiar desviarse, entonces el NE no es robusto a ellas.

## Juegos de coordinación

*   Los juegos de coordinación poseen más de un NE. Hay casos donde hay NEs mejores que otros para todos los jugadores, otros donde todos los NEs son iguales, y otros donde cada jugador prefiere un NE distinto.
*   Dependiendo de las creencias de los jugadores, es posible que se llegue a un NE "malo". Pero si los jugadores se comunican y coordinan, pueden moverse hacia un NE "bueno" y así salir ganando todos.
*   Es un caso distinto al dilema del prisionero porque la comunicación puede hacer que se pase de un NE a otro, pero no puede hacer que se elija una estrategia estrictamente dominada.

## Estrategias mixtas

*   Una estrategia mixta pᵢ es una elección aleatoria entre las posibles estrategias puras sᵢ. Esto puede convenir cuando no existe un NE usando solo estrategias puras o cuando los jugadores prefieren distintos NE. Por ejemplo, pᵢ = (1/2, 1/2, 0) 
*   Una estrategia pura es un caso especial de una estrategia mixta, en donde una única estrategia tiene una probabilidad de 1 y el resto de 0.
*   Cuando se juega una estrategia mixta, la recompensa esperada es un promedio ponderado de las recompensas esperadas de cada una de las estrategias puras. Esto significa que la recompensa esperada siempre se encuentra en algún punto entre la mayor y la menor recompensa posible.
*   La condición para que un perfil de estrategias mixtas p' sea un NE es la misma que antes: cada pᵢ' es mejor respuesta con respecto a p₋ᵢ'. 
*   Si una estrategia mixta es una mejor respuesta, entonces cada una de las estrategias involucradas (con probabilidad mayor a cero) también tienen que ser mejores respuestas. 
*   Combinando los dos puntos anteriores se deriva que, para que un mix sea NE, las recompensas esperadas para cada una de las estrategias de un jugador tienen que ser iguales (si no lo fueran se descartarían las estrategias que llevan a menores recompensas). Estas recompensas dependen de las probabilidades de los demás, no de las propias.
*   Hay tres posibles interpretaciones acerca de las probabilidades en una estrategia mixta: aleatorización (depende del azar), creencias (depende de lo que uno cree que los demás van a hacer) y proporciones (depende de las diferencias entre subgrupos de una población).

## Evolución

*   En un contexto biológico, se pueden asociar a las estrategias con genes y a las recompensas con aptitud genética. Dichas estrategias no son elegidas por individuos racionales sino que están "cableadas" biológicamente.
*   Una estrategia crece si le va bien (la población con el gen continúa reproduciéndose) o muere si le va mal (la población con el gen se extingue). Lo que importa es la supervivencia del gen, no la del individuo.
*   En un juego simétrico de dos jugadores, una estrategia s' (ya sea pura o mixta) es evolutivamente estable (ES) si se cumplen dos condiciones:
    - 1. El perfil (s', s') es un NE (u(s',s') ≥ u(s,s') para todo s).
    - 2. Si u(s',s') = u(s,s') para algún s, entonces se tiene que dar que u(s',s) > u(s,s) (la estrategia original debe ser mejor contra la invasora que la invasora contra sí misma).
*   En el dilema del prisionero, cooperar no es una estrategia ES pero desertar sí lo es.

## Juegos secuenciales

*   Que un juego sea simultáneo o secuencial no depende del flujo del tiempo, sino del flujo de información.
*   Backward induction es el proceso de determinar la secuencia optima de acciones (en cada punto de un árbol) al razonar desde el punto final de un juego. Funciona eliminando las amenazas que no son creíbles.  
*   En algunos casos, tener más información, más opciones o mayores recompensas en juegos secuenciales puede ser contraproducente. Por ejemplo, que un rival sepa que uno tiene cierta información puede llevarlo a realizar acciones que te impacten negativamente.
*   Un juego es de información perfecta si, para cada nodo, el jugador que le toca su turno sabe en que nodo se encuentra. En este tipo de juego, una estrategia es un plan de acción completo que especifica qué acción se debe elegir en cada nodo.
*   Teorema de Zermelo: cualquier juego de dos jugadores con información perfecta, nodos finitos y tres posibles resultados (victoria, derrota o empate) puede ser resuelto. Esto quiere decir que, suponiendo que ambos jugadores juegan a la perfección, siempre alguno puede forzar una victoria o un empate.
*   La paradoja de la cadena de tiendas enseña que a veces conviene tomar acciones que parecen no racionales para establecer una reputación que te ayude a largo plazo.
*   Juego de negociación con ofertas alternadas: si se asume que una negociación puede ser eterna, que las ofertas se pueden hacer muy rápido (el factor de descuento es cercano a uno) y que ambos jugadores tienen el mismo factor de descuento (son igual de impacientes; sopesan el futuro y el presente de la misma manera), entonces la repartición de la diferencia va a tender a ser repartida mitad y mitad directamente en la primera oferta. En un caso más realista, esto último no se cumple porque uno no puede saber exactamente cuál es el factor de descuento de los demás ni cuál es el valor que asignan al objeto por el que se negocia.

## Juegos de información imperfecta

*   Que en un juego haya información imperfecta significa que hay casos donde un jugador no sabe en qué nodo del árbol del juego se encuentra. Estos nodos indistinguibles forman un set de información, y en ellos no se puede aplicar backward induction.
*   Un juego simultáneo puede pensarse como un juego secuencial con información imperfecta: jugar a la vez es lo mismo que no saber lo que el rival jugó; lo importante no es el tiempo sino la información.
*   Un subjuego cumple tres condiciones: empieza en un único nodo, contiene todos los nodos sucesores del inicial, y no rompe ningún set de información.
*   Un NE es perfecto en subjuegos (SPNE) si induce un NE en cada subjuego existente.

## Interacciones repetidas

*   En una relación que se mantiene a lo largo del tiempo, la promesa de recompensas futuras y/o la amenaza de futuros castigos a veces puede incentivar a la cooperación. Pero para que esto funcione, justamente tiene que haber un futuro: la relación no puede tener un final premeditado.
*   Mientras más peso tenga el futuro (que puede estar relacionado con la importancia, paciencia o probabilidad de que exista), es más fácil que los incentivos para cooperar superen a las tentaciones de desertar.
*   En un juego repetido indefinidamente, casi cualquier resultado que sea mejor para todos los jugadores que el peor castigo posible puede ser sostenido como un NE, siempre y cuando los jugadores sean lo suficientemente pacientes. 

## Torneos de Axelrod

*   La cooperación puede emerger y sostenerse aún cuando los jugadores solo están motivados por su interés propio; el altruismo no es necesario. 
*   Tit for Tat y las demás estrategias con buenos desempeños poseen cuatro características:
    - 1. Bondad: cooperan por defecto.
    - 2. Perdón: no dejan que las rondas anteriores a la última influencien las decisiones actuales.
    - 3. Provocabilidad: no se dejan pasar por encima, toman represalias inmediatamente.
    - 4. Claridad: es posible entenderlas y  establecer un patrón de confianza con ellas.
*   No existe una estrategia óptima, siempre la mejor elección depende de las estrategias con las que se vaya a interactuar.
*   Como un ambiente realista es ruidoso (existe una pequeña probabilidad de que una cooperación sea percibida como una deserción y visceversa), TFT tiene la limitación de permitir bucles infinitos de represalias. 
*   Algunas características que incorporan estrategias más complejas son: tolerancia (aceptar que los errores existen y perdonarlos), memoria (recordar estados previos al último), adaptación (cambiar de comportamiento de acuerdo al oponente) y contextualidad (formar alianzas y castigar a oportunistas).