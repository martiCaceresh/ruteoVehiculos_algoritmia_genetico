from os import remove
import numpy as np
import copy
import sys
from numpy import loadtxt

distancias=[[0,101,42,76,24,38,89,110],
	        [101,0,107,39,77,69,92,33],
	        [42,107,0,72,52,38,97,116],
	        [76,39,72,0,86,38,131,44],
	        [24,77,52,86,0,48,65,86],
	        [38,69,38,38,48,0,93,78],
	        [89,92,97,131,65,93,0,125],
	        [110,33,116,44,86,78,125,0]]

pasajeros=[[0,9,6,10,4,54,21,51],
           [28,0,16,44,4,41,4,24],
	       [26,51,0,50,14,56,16,1],
	       [39,14,12,0,25,40,52,8],
	       [19,55,60,42,0,37,32,37],
	       [6,10,59,36,60,0,18,10],
	       [35,37,25,5,32,9,0,3],
	       [43,4,27,17,6,42,9,0]]


def buscar(nodo,nodos_totales):
	size=len(nodos_totales)
	i=0
	while i<size:
		if nodos_totales[i]==nodo:
			nodos_totales.remove(nodo)
			return nodos_totales
		i+=1

	return nodos_totales

def barrio_aislado(linea,nodos_totales):
	for i in range(len(linea)): #recorro los nodos o la zonas que recorro en la ruta
		nodos_totales=buscar(linea[i],nodos_totales)

	return nodos_totales

def iguales(ruta1,ruta2):
	size1=len(ruta1)
	if size1!=len(ruta2):
		return False

	else:
		for i in range(size1):
			if ruta1[i]!=ruta2[i]:
				return False


	return True


def rutas_repetidas(individuo):
	size=len(individuo)
	for i in range(size-1):
		temp=individuo[i]
		for j in range(i+1,size):
			if iguales(temp,individuo[j]):
				return True

	return False

def repetidos(ruta,nodo):
	if len(ruta)==0:
		return False

	for i in range(len(ruta)):
		if nodo==ruta[i]:
			return True

	return False


def conexion(ruta1,ruta2): # devuelve true si existe algun nodo en comun
	for i in range(1,len(ruta1)-1): #recorro todos los nodos excepto el primer nodo
		for j in range(1,len(ruta2)-1): #recorro todos los nodos excepto el primero
			if ruta1[i]==ruta2[j]:
				return True

	return False

def ruta_aislada(individuo):
	size=len(individuo) #cantidad de rutas que tiene
	for i in range(size-1): #recorro las rutas del individuo menos 1 
		for j in range(i+1,size): #recorro las rutas a partir de la ruta indexada con i
			if conexion(individuo[i],individuo[j]):
				return True #retorna true si existe alguna ruta conectada
	return False #retorna false si no tiene rutas conectadas

def individuo_valido(individuo):
	nodos_totales=copy.copy(nodos) #copio los nodos en una lista de nodos_totales
	for i in range(len(individuo)): #recorro todas las rutas del individuo
		nodos_totales=barrio_aislado(individuo[i],nodos_totales) #le voy quitando todos los nodos que recorro con las rutas de mi individuo

	ra=not ruta_aislada(individuo) #el individuo es valido cuando tiene todas sus rutas aisladas
	ba=not len(nodos_totales) # el individuo es valido cuando recorre todo el espacio 
	rp=not rutas_repetidas(individuo) # el individuo es valido cuando no tiene rutas repetidas

	return ba and ra and rp

def generar_ruta():
	ruta=[]
	size_ruta=np.random.randint(nodos_minimo,n_nodos) # al azar se escoge el tamaño que va a tener una ruta
	temp=copy.copy(nodos) # se copia la matriz de nodos en una variable temporal
	for j in range(size_ruta):
		nodo=np.random.choice(temp) #se escoge un nodo al azar para formar la ruta
		temp.remove(nodo) #se remueve el nodo del arreglo de nodos para que ya no sea escogido
		ruta.append(nodo) # se agrega el nodo recorrido
	return ruta
def buscaUltimo(linea):
	for i in range(len(linea)):
		ultimo = i
	return linea[ultimo]

def escogerMejorRutaInsertar(lineas,nodo):
	valorMax = 0
	mejorRuta = -1#inicializo con una lista vacia
	for i in range(len(lineas)):
		nodoIni = buscaUltimo(lineas[i])
		atencion = pasajeros[nodoIni][nodo] # calculo la cantidad de pasajeros que atiendo
		distancia = distancias[nodoIni][nodo] #calculo la distancia recorrida
		valor = atencion/distancia
		if valor > valorMax and valor > 0.5 : 
			mejorRuta = i #almaceno la linea de mayor valor

	return mejorRuta
def insertarNodo(linea,nodo):
	linea.append(nodo)

def generarRuta(nodo):
	linea=[0,nodo]
	return linea

def generar_individuo():
	lineas=[]
	temp = copy.copy(nodos)
	temp.remove(0)
	for i in range(len(temp)):
		nodo = np.random.choice(temp) #se escoge un nodo al azar
		mejorLinea = escogerMejorRutaInsertar(lineas,nodo) #funcion que regresa la mejor linea para insertar
		temp.remove(nodo)
		if mejorLinea != -1 :
			insertarNodo(lineas[mejorLinea],nodo)
		elif len(lineas) <=  max_rutas :
			lineas.append(generarRuta(nodo))
	return lineas, individuo_valido(lineas)

def fitness(individuo,m_distancias):
	temp=[]
	for i in range(n_nodos):
		temp.append(copy.copy(m_distancias[i]))

	for i in range(len(individuo)):
		for j in range(len(individuo[i])-1):
			temp[individuo[i][j]][individuo[i][j+1]]=temp[individuo[i][j]][individuo[i][j+1]]+5
			temp[individuo[i][j+1]][individuo[i][j]]=temp[individuo[i][j]][individuo[i][j+1]]

	fit=0.0
	valor=0.0
	for i in range(n_nodos):
		for j in range(n_nodos):
			if i!=j:
				valor=(1.0*(pasajeros[i][j]))/temp[i][j]
				fit=fit+valor

	return fit


def calc_fitness(individuo,f_e):
	fit=fitness(individuo,m_distancias)
	num_lineas=(1.0*len(individuo))/(max_rutas-min_rutas)
	es_valido=individuo_valido(individuo)
	penalizacion_inv=0
	if not es_valido:
		penalizacion_inv=1000000

	optimizacion=fit-fit*f_e*num_lineas-penalizacion_inv
	return optimizacion


def generar_poblacion(size):
	print ("************************************ Primera Poblacion *************************************")
	poblacion=[]
	fit_poblacion=[]
	i=0
	while i<size: #vamos a generar una cantidad determinada de individuos (25)
		individuo,valido=generar_individuo() #algoritmo PIA
		while not valido:
			mutacion_individual(individuo)
			valido=individuo_valido(individuo)

		poblacion.append(individuo)
		fit_poblacion.append(calc_fitness(individuo,f_ecologia))
		print (i+1,"Individuo Aceptado:",poblacion[i], "fit", fit_poblacion[i])
		i+=1

	return poblacion, fit_poblacion

def mutar():
	nueva_ruta=[]
	longitud_ruta=np.random.randint(nodos_minimo,n_nodos)
	lista_nodos=copy.copy(nodos)
	for i in range(longitud_ruta):
		nodo=np.random.choice(lista_nodos)
		lista_nodos.remove(nodo)
		nueva_ruta.append(nodo)

	return nueva_ruta

def mutacion(individuo,long_original):
	nueva_longitud=np.random.randint(min_rutas,max_rutas)
	if nueva_longitud<long_original:
		for i in range(nueva_longitud):
			aleatorio=np.random.uniform(0,1)
			if aleatorio<pm:
				individuo[i]=mutar()
			del individuo[nueva_longitud]
		else:
			j=0
			for i in range(nueva_longitud):
				if j<long_original:
					aleatorio=np.random.uniform(0,1)
					if aleatorio<pm:
						individuo[i]=mutar()
					j+=1
				else:
					individuo.append(mutar())

def mutacion_individual(individuo):
	longitud=len(individuo)
	aleatorio=np.random.uniform(0,1)
	if aleatorio<pm:
		print ("Individuo:",individuo, "fit:", calc_fitness(individuo,f_ecologia))
		mutacion(individuo,longitud)
		print ("Individuo Mutado:",individuo, "fit:", calc_fitness(individuo,f_ecologia))


def mutacion_poblacion(poblacion):
	for i in range(n_individuos):
		mutacion_individual(poblacion[i])


def crear_grafo(nodos):
	grafo = [[0 for columna in range(nodos)
					for fila in range(nodos)]]
	return grafo

def imprimir(dist):
            print ([dist[nodo] for nodo in range(n_nodos)])

def distancia_min(dist, nv):

        minimo = 100000000

        for n in range(n_nodos):
            if dist[n] < minimo and nv[n] == False:
                minimo = dist[n]
                min_indice = n

        return min_indice

def dijkstra(inicio,grafo,matriz_dist):
	dist = [100000000]*n_nodos
	dist[inicio] = 0
	nv = [False]*n_nodos
	for i in range(n_nodos):
		a=distancia_min(dist,nv)
		nv[a]=True
		for b in range(n_nodos):
			if grafo[a][b] > 0 and nv[b] == False and dist[b] > dist[a] + grafo[a][b]:
				dist[b] = dist[a] + grafo[a][b]
	matriz_dist.append(dist)

def iniciar_dijkstra(grafo):
	matriz_dist=[]
	for i in range(n_nodos):
		dijkstra(i,grafo,matriz_dist)

	return matriz_dist

def torneo(poblacion,participantes,fit_poblacion):
	indice_part=[]
	for i in range(participantes):
		indice_part.append(np.random.randint(n_individuos))

	maximo=-1000000000
	for i in indice_part:
		if fit_poblacion[i]>maximo:
			maximo=fit_poblacion[i]
			indice_win=i

	return indice_win

def cruce(individuo1,individuo2):
	hijo=[]
	long1=np.random.randint(len(individuo1))
	long2=np.random.randint(len(individuo2))
	while long1+long2<2:
		long1=np.random.randint(len(individuo1))
		long2=np.random.randint(len(individuo2))

	ind_elegido1=copy.copy(individuo1)
	ind_elegido2=copy.copy(individuo2)

	for i in range(long1):
		nodo=np.random.choice(ind_elegido1)
		ind_elegido1.remove(nodo)
		hijo.append(nodo)

	for i in range(long2):
		nodo=np.random.choice(ind_elegido2)
		ind_elegido2.remove(nodo)
		if not repetidos(hijo,nodo):
			hijo.append(nodo)

	if len(hijo)==1:
		temp=copy.copy(nodos)
		temp.remove(hijo[0])
		nodo=np.random.choice(temp)
		hijo.append(nodo)

	return hijo

def cruzar(individuo1,individuo2):
	aleatorio=np.random.uniform(0,1)
	if aleatorio<pc:
		print ("Individuo elegido:", individuo1, "fit:",calc_fitness(individuo1,f_ecologia))
		print ("Individuo elegido:", individuo2, "fit:",calc_fitness(individuo2,f_ecologia))
		pos_cruce1=np.random.randint(len(individuo1))
		pos_cruce2=np.random.randint(len(individuo2))
		print ("Cruzamiento en:", individuo1[pos_cruce1])
		print ("Cruzamiento en:", individuo2[pos_cruce2])
		temp=cruce(individuo1[pos_cruce1],individuo2[pos_cruce2])
		print ("Cruce:", temp)
		individuo2[pos_cruce2]=cruce(individuo1[pos_cruce1],individuo2[pos_cruce2])
		print ("Cruce:",individuo2[pos_cruce2])
		individuo1[pos_cruce1]=temp
		print ("Individuo cruzado:", individuo1, "fit:",calc_fitness(individuo1,f_ecologia))
		print ("Individuo cruzado:", individuo2, "fit:",calc_fitness(individuo2,f_ecologia))
		print


def cruzamiento(poblacion,fit_p,participantes):
	temp=[]
	for i in range(n_individuos):
		ind=[]
		for j in range(len(poblacion[i])):
			ind.append(copy.copy(poblacion[i][j]))

		temp.append(ind)

	for i in range(n_individuos/2):
		ip1=torneo(temp,participantes,fit_p)
		ip2=torneo(temp,participantes,fit_p)
		cruzar(poblacion[ip1],poblacion[ip2])

	return poblacion

def validar_poblacion(poblacion):
	fit_poblacion=[]
	print
	print ("********************************* Poblacion Valida *************************************")
	for i in range(n_individuos):
		valido=individuo_valido(poblacion[i])
		while not valido:
			mutacion_individual(poblacion[i])
			valido=individuo_valido(poblacion[i])

		fit_poblacion.append(calc_fitness(poblacion[i],f_ecologia))
		print (i+1, poblacion[i], "fit:",fit_poblacion[i])

	return poblacion,fit_poblacion


def algorimo_genetico(n_individuos,f_ecologia,n_participantes,n_iteraciones):

	for i in range(n_iteraciones):
		print
		print ("************************************* Iteracion:",i+1,"**************************************")
		print
		print ("************************************* Cruzamiento **************************************")

		poblacion=cruzamiento(poblacion,fit,n_participantes)
		print
		print ("************************************* Mutacion **************************************")
		mutacion_poblacion(poblacion)
		poblacion,fit_p=validar_poblacion(poblacion)
		fit=fit_p
		ordenar=list(np.argsort(fit_p))
		ordenar.reverse()
		print
		print (ordenar[0]+1,"Mejor:",poblacion[ordenar[0]], fit_p[ordenar[0]])

	print
	print ("***************************** Ultima Poblacion *********************************")
	for i in range(n_individuos):
		print (i+1,poblacion[i], "fit: ",fit_p[i])
	print
	print (ordenar[0]+1,"Mejor Solucion:",poblacion[ordenar[0]], fit_p[ordenar[0]])



n_nodos=len(distancias)
nodos=range(n_nodos)
nodos=list(nodos)
grafo=crear_grafo(n_nodos)
grafo = distancias
m_distancias=iniciar_dijkstra(grafo)

max_rutas=n_nodos-1
min_rutas=2
nodos_minimo=2
n_individuos=25
f_ecologia=0
n_participantes=3
pm=0.3
pc=0.5
n_iteraciones=50
valores=[]
for i in range(n_nodos):
	for j in range(n_nodos):
		if i != j : 
			valor = pasajeros[i][j] / distancias[i][j]
			valores.append(valor)

print
poblacion,fit=generar_poblacion(n_individuos)
#algorimo_genetico(n_individuos,f_ecologia,n_participantes,n_iteraciones,poblacion,fit)

