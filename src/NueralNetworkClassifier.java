import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;





/*LAST THING I WAS DOING WAS TESTING DELTAB WITH PEER*/

/* Point - object representing multi-dimensional data point*/
class Point{
	double target[];
	double values[];

	public Point(double vals[], double target[]){
		this.values = vals;
		this.target = target;
	}
	public Point(double vals[]){
		this.values = vals;
		this.target = null;
	}

	public void setTarget(double target[]){
		this.target = target;
	}
}

public class NueralNetworkClassifier {
	
	/*Globals*/
	private static int MAX_BAD_COUNT = 20;
	private static int EPOCHS = 20;
	private static boolean VERBOSE = false;
	private static boolean QUIET = false;
	private static boolean DEBUG = false;
	private static NumberFormat itersFormat = new DecimalFormat("#0000");
	private static NumberFormat objFormat = new DecimalFormat(".#####");
	private static NumberFormat yValFormat = new DecimalFormat(".##");
	
	/*From config file*/
	private static int N_TRAIN = -1;
	private static int N_DEV = -1;
	private static String TRAIN_X_FN = null;
	private static String TRAIN_T_FN = null;
	private static String DEV_X_FN = null;
	private static String DEV_T_FN = null;
	private static int D = -1;
	private static int C = -1;
	
	/*From command line args*/
	private static int L=-1;
	private static double ss=-1;
	private static int mb=-1;
	private static double alpha=-1;
	
	/*From data files*/
	private static Point[] train;
	private static Point[] dev;

	/*./NueralNetworkClassifier -task c|r|l [-tanh] [-standard] [-batch mb] [-mo alpha] L ss data_cfg_fn*/
	//Reads command line args, starts correct task
	public static void main(String[] args){
		
		//int L = -1; //hidden layer nodes
		//double ss = -1; //step size
		String cfg_file=null; //name of data config file
		String task=null;
		boolean tanh=false,standard=false,batch=false,momentum=false;
		//int mb = -1; //batch size
		//double alpha = -1; //for momentum

		for (int i=0; i<args.length; i++){
			if (args[i].equals("-task")){
				if (args[i+1].equals("c") || args[i+1].equals("r") || args[i+1].equals("l")){
					task = args[i+1];
					i++;
				} else {
					System.err.println("unknown task, please use c,r,l");
					System.exit(-1);
				}	
			} else if (args[i].equals("-tanh")){
				tanh = true;
			} else if (args[i].equals("-standard")){
				standard = true;
			} else if (args[i].equals("-batch")){
				mb = Integer.valueOf(args[i+1]);
				batch = true;
				i++;
			} else if (args[i].equals("-mo")){
				alpha = Double.valueOf(args[i+1]);
				momentum = true;
				i++;
			} else if (args[i].equals("-bc")){
				MAX_BAD_COUNT = Integer.valueOf(args[i+1]);
				i++;
			} else if (args[i].equals("-epochs")){
				EPOCHS = Integer.valueOf(args[i+1]);
				i++;
			} else if(args[i].equals("-v")){
				VERBOSE = true;
			} else if(args[i].equals("-q")){
				QUIET = true;
			} else if(args[i].equals("-DEBUG")){
				DEBUG = true;
			} else {
				L = Integer.valueOf(args[i]);
				ss = Double.valueOf(args[i+1]);
				cfg_file = args[i+2];
				break;
			}	
		}
		
		readInputFiles(cfg_file);
		
		/*for (int n=0; n<N_DEV; n++){
			for (int c=0; c<C; c++){
				System.out.print(dev[n].target[c] + ",");
			}
			System.out.print(" : ");
			for (int d=0; d<D+1; d++){
				System.out.print(dev[n].values[d] + ",");
			}
			System.out.println();
		}*/
		
		backProp(tanh,standard,task);
		/*if (task.equals("c")){
			classification(tanh,standard);
		} else if (task.equals("r")){
			regression(tanh,standard);
		} else if (task.equals("l")){
			logistic(tanh,standard);
		}*/
		
	}
	
	public static void backProp(boolean tanh, boolean standard, String task){
		
		/*initialize W(D+1xL) and U(L+1xC)*/
		double[][] W = new double[L][D+1];
		double[][] U = new double[L+1][C];
		
		Random randW = new Random();
		double maxW = -1/(Math.sqrt(D+1));
		double minW = -1 * maxW;
		
		Random randU = new Random();
		double maxU = -1/(Math.sqrt(L+1));
		double minU = -1*maxU;
		
		for (int d=0; d<D+1; d++){
			for (int l=0; l<L; l++){
				W[l][d] = minW + (maxW - minW) * randW.nextDouble();
				if(DEBUG){
					W[l][d] = .5;
				}
			}
		}
		
		for (int l=0; l<L+1; l++){
			for (int c=0; c<C; c++){
				U[l][c] = minU + (maxU - minU) * randU.nextDouble();
				if(DEBUG){
					U[l][c] = .5;
				}
			}
		}
		
		int iters=0,batch_size =0,batch_count=0, epochs=0;
		int bad_count=0;
		double best_dev_loss =Double.MAX_VALUE;
		boolean converged = false;
		double train_loss=0,dev_loss=0;
		double[][] gradW = new double[L][D+1];
		double[][] gradU = new double[L+1][C];
		double[][] dev_guesses = new double[N_DEV][C];
		if (mb>0){
			batch_size = mb;
		}else{
			batch_size = 1;
		}
		
		/*-------main loop-------*/
		while (!converged && epochs < EPOCHS){
			for(int n=0;n<N_TRAIN;n++){
				
			/*zero*/
				double[] H = new double[L+1];
				double[] Y = feedForward(task,train[n], W, U, H);
				
				if(DEBUG){
					System.out.print("W : \n");
					for (int l=0;l<L;l++){
						System.out.print(Arrays.toString(W[l]));
						System.out.print("\n");
					}
					System.out.print("U : \n");
					for (int l=0; l<L+1; l++){
						System.out.print(Arrays.toString(U[l]));
						System.out.print("\n");
					}
				}
				
			/*one*/	
				/*our guesses are now found, so now we start backprop*/
				/*calc Delta B = t-y*/
				
				if(DEBUG){System.out.print("train[" + n + "]\n deltaB : ");}
				double[] deltaB = new double[C];
				for (int c=0;c<C;c++){
					deltaB[c] = train[n].target[c] - Y[c];
					if(DEBUG){System.out.print(deltaB[c] + " ");}
				}
				if(DEBUG){System.out.print("\n");}
			/*two*/
				/*calc gradient of U*/
				/*negate H and do outer product with deltaB*/
				double[] negH = new double[L+1];
				for (int l=0; l<L+1; l++){
					negH[l] = -1*H[l];
				}
				gradU = matrixAdd(gradU,outerProduct(negH,deltaB));
				if(DEBUG){
					System.out.print("gradU : \n");
					for (int l=0;l<L+1;l++){
						System.out.print(Arrays.toString(gradU[l]));
						System.out.print("\n");
					}
				}
				
				
			/*three*/	
				/*throw away bias! calc deltaA*/
				double[] deltaA = new double[L];
				if(DEBUG){
					System.out.print("H = ");
					for (int l=0;l<L+1;l++){
						System.out.print(H[l] + " ");
					}
				}
				if(DEBUG){System.out.print("\nfprime = ");}
				double[] fprime = new double[L];
				for (int l=0;l<L;l++){
					fprime[l] = H[l+1] * (1-H[l+1]);//this is l+1 so we throw away bias
					if(DEBUG){System.out.print(fprime[l] + "  ");}
				}
				if(DEBUG){System.out.print("\n");}

				
				//throw away bias on U
				
				double[][] noBiasU = new double[L][C];
				for (int c=0;c<C;c++){
					for (int l=1;l<L+1;l++){
						noBiasU[l-1][c]=U[l][c];
					}
					

				}
				if(DEBUG){
					System.out.print("noBiasU = \n");

					for (int l=0;l<L;l++){
						System.out.print(Arrays.toString(noBiasU[l]));
						System.out.print("\n");
					}
				}
				//calc U outer product deltaB
				double[] product = new double[L];
				product = matrixByVector(noBiasU,deltaB);
				if(DEBUG){
					System.out.print("noBiasU = \n");
					System.out.print("noBiasU x deltaB = ");
					for (int l=0; l<L; l++){
						System.out.print(product[l] + " ");
					}
					System.out.print("\n");
				}

				//elementwise multiply product with fprime
				if(DEBUG){System.out.print("deltaA : ");}
				for (int l=0; l<L; l++){
					deltaA[l] = product[l] * fprime[l];
					if(DEBUG){System.out.print(deltaA[l] + " ");}
				}
				if(DEBUG){System.out.print("\n");}
				
			/*four*/
				/*calc gradient of W*/
				//double[][] gradW = new double[L][D+1];
				//negate datapoint, outer product with deltaA
				double[] negTrain = new double[D+1];
				for (int d=0;d<D+1;d++){
					negTrain[d] = train[n].values[d] * -1;
				}
				gradW = matrixAdd(gradW,outerProduct(deltaA,negTrain));
				if(DEBUG){
					System.out.print("gradW = \n");
					for (int l=0;l<L;l++){
						System.out.print(Arrays.toString(gradW[l]));
						System.out.print("\n");
					}

					System.out.print("Y's before grad added : ");
					for (int c=0;c<C;c++){
						System.out.print(Y[c] + " ");
					}
					System.out.print("\n");
				}
				
				batch_count++;
				if (batch_count == batch_size ){
					batch_count = 0;
					/*five*/
						/*update U & W w/ gradient descent*/
						for (int l=0;l<L;l++){
							for (int d=0;d<D+1;d++){
								W[l][d]-=ss*gradW[l][d];
								gradW[l][d] = 0;
							}
						}
						for (int l=0;l<L+1;l++){
							for (int c=0;c<C;c++){
								U[l][c]-= ss * gradU[l][c];
								gradU[l][c] = 0;
							}
						}
						
						
					/*try dev set with new U and W*/
						dev_loss = 0;
						//double[][] dev_guesses = new double[N_DEV][C];
						for (int i=0; i<N_DEV; i++){
							double[] devY = new double[C];
							double[] devH = new double[L+1];//kinda un-needed

							devY = feedForward(task,dev[i],W,U,devH);
							dev_guesses[i] = devY;
							if (task.equals("c")){
								dev_loss -= classLoss(dev[i],devY);
							} else if(task.equals("r")){
								dev_loss += regLoss(dev[i],devY);
							} else if(task.equals("l")){
								dev_loss -= logLoss(dev[i],devY);
							}
						}
						//dev_loss=dev_loss/N_DEV;
						
						train_loss = 0;
						double[][] train_guesses = new double[N_TRAIN][C];
						for (int i=0; i<N_TRAIN; i++){
							double[] trainY = new double[C];
							double[] trainH = new double[L+1];//kinda un-needed

							trainY = feedForward(task,train[i],W,U,trainH);
							train_guesses[i] = trainY;
							if (task.equals("c")){
								train_loss -= classLoss(train[i],trainY);
							} else if(task.equals("r")){
								train_loss += regLoss(train[i],trainY);
							} else if(task.equals("l")){
								train_loss -= logLoss(train[i],trainY);
							}
						}						
						
						/*early stop calcs*/
						if (dev_loss>best_dev_loss){
							bad_count++;
						} else {
							best_dev_loss = dev_loss;
						}

						if (bad_count > MAX_BAD_COUNT){
							converged = true;
							break;
						}
						
						/*print*/
						
						if (!QUIET){
							String itersString = itersFormat.format(iters);
							String devObjString = objFormat.format(dev_loss/N_DEV);
							
							System.err.print("Iter " + itersString + ": devObj=" + devObjString + "\n");
							if (VERBOSE){
								printGuesses(train_guesses,N_TRAIN,C,"train");
							}
							printGuesses(dev_guesses,N_DEV,C,"dev");
						}

						
						
						/*DEBUG*/
						if(DEBUG){
							System.out.print("train["+n+"]\n");
							System.out.print("target = ");
							for (int c=0;c<C;c++){
								System.out.print(train[n].target[c] + " ");
							}
							System.out.print("\n");
							System.out.print("Ys = ");
							for (int c=0;c<C;c++){
								String temp = yValFormat.format(train_guesses[n][c]);
								System.out.print(temp + " ");
							}
							System.out.print("\n");
							System.out.print("\n");
						}
						iters++;
				}
			}
			if(QUIET){
				String itersString = itersFormat.format(iters);
				String devObjString = objFormat.format(dev_loss);
				System.err.print("Iter " + itersString + ": devObj=" + devObjString + "\n");
				printGuesses(dev_guesses,N_DEV,C,"dev");
			}
			epochs++;
		}
		String trainObjString = objFormat.format(train_loss/N_TRAIN);
		String devObjString = objFormat.format(dev_loss/N_DEV);
		
		System.err.print("Final (iters="+iters+") " + "trainObj=" + trainObjString + " devObj=" + devObjString + "\n");
		
	}
	private static void printGuesses(double[][] Y, int N, int C, String name){
		System.out.print(name + " ");
		for (int i=0;i<N;i++){
			System.out.print("[");
			for (int j=0;j<C;j++){
				if(j!=0){
					System.out.print(" ");
				}
				String yVal = yValFormat.format(Y[i][j]);
				System.out.print(yVal);
			}
			System.out.print("] ");
		}
		System.out.print("\n");
	}
	private static double classLoss(Point p, double[] Y){
		double sum=0;
		for (int c=0;c<C;c++){
			sum += p.target[c] * Math.log(Y[c]);
		}
		return sum;
	}
	private static double regLoss(Point p, double[] Y){
		double sum=0;
		for (int c=0;c<C;c++){
			sum += Math.pow(p.target[c] - Y[c],2);
		}
		return sum;
	}
	private static double logLoss(Point p, double[] Y){
		double sum=0;
		for (int c=0;c<C;c++){
			double first = p.target[c] * Math.log(Y[c]);
			double second = (1-p.target[c]) * Math.log(1-Y[c]);
			sum += first + second;
		}
		return sum;
	}
	
	private static double binLogClassify(double[] X, double[] betas){
		Sigmoid sig = new Sigmoid();
		return sig.value(innerProduct(X, betas));
	}
	
	private static double[] feedForward(String task, Point point, double[][] W, double[][] U, double[] H){

		H[0] = 1;
		for (int l=1; l<L+1; l++){
			H[l] = binLogClassify(point.values,W[l-1]);
		}
		
		
		/*multinomial log regression on each training datapoint*/
		double[] B = new double[C];
		for (int c=0; c<C; c++){
			double[] colU = new double[L+1];
			for (int l=0;l<L+1;l++){
				colU[l] = U[l][c];
			}
			B[c] = innerProduct(colU,H); 
		}
		
		if (task.equals("c")){
			double denom=0;
			for(int c=0;c<C;c++){
				denom += Math.exp(B[c]);/*--------------!Double check this!!!-------------------*/
			}
			double[] Y = new double[C];
			for (int c=0;c<C;c++){
				Y[c] = Math.exp(B[c])/denom;
			}
			return Y;
		} else if (task.equals("r")){
			return B;
		}else if (task.equals("l")){
			double[] Y = new double[C];
			Sigmoid sig = new Sigmoid();
			for (int c=0;c<C;c++){	
				Y[c] = sig.value(B[c]);
			}
			return Y;
		}
		return null;
	}
	
	
	//reads the config file, calls functions that read the data and target files
	private static void readInputFiles(String dataConfig){
		try{
			FileInputStream fstream = new FileInputStream(dataConfig);
			DataInputStream in = new DataInputStream(fstream);
			@SuppressWarnings("resource")
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			
			/* config file parsing */
			while ((strLine = br.readLine()) != null){
				String[] configStrings = strLine.split("\\s+");
				switch ( configStrings[0]){
				case "N_TRAIN"://REQUIRED
					N_TRAIN = Integer.valueOf(configStrings[1]);
					break;
				case "N_DEV"://REQUIRED
					N_DEV = Integer.valueOf(configStrings[1]);
					break;
				case "TRAIN_X_FN"://REQUIRED
					TRAIN_X_FN = configStrings[1];
					break;
				case "TRAIN_T_FN"://REQUIRED 
					TRAIN_T_FN = configStrings[1];
					break;
				case "DEV_X_FN"://REQUIRED 
					DEV_X_FN = configStrings[1];
					break;
				case "DEV_T_FN"://REQUIRED 
					DEV_T_FN = configStrings[1];
					break;
				case "D"://REQUIRED 
					D = Integer.valueOf(configStrings[1]);
					break;
				case "C"://REQUIRED 
					C = Integer.valueOf(configStrings[1]);
					break;
				}
			}
		}catch(Exception e){
			System.err.println("1");
			System.err.println("Error: " + e.getMessage());
			return;
		}
		
		//Check if config file had everything required
		if(N_TRAIN<1 || N_DEV<1 || D<1 || C<1 ||
				TRAIN_X_FN==null || TRAIN_T_FN==null || DEV_X_FN==null ||DEV_T_FN==null){
			System.err.println("Missing values in config file -- terminating");
			System.exit(-1);
		}
		
		train = loadData(TRAIN_X_FN,N_TRAIN,D);
		dev = loadData(DEV_X_FN,N_DEV,D);
		
		setTargets(train,N_TRAIN,D,TRAIN_T_FN);
		setTargets(dev,N_DEV,D,DEV_T_FN);
	}
	
	//reads data files
	private static Point[] loadData(String dataFile, int N, int D){
		Point[] data = new Point[N];
		try{
			@SuppressWarnings("resource")
			FileInputStream fstream = new FileInputStream(dataFile);
			DataInputStream in = new DataInputStream(fstream);
			@SuppressWarnings("resource")
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			
			int i=0,j=1,z=0;
			while(((strLine = br.readLine()) != null) && i< N){
				String[] valueStrings = strLine.split(" ");
				if(valueStrings.length != D){
					System.err.println("Not enough doubles on line " + i + " of the data set in " + dataFile);
					return null;
				}
				double[] valueDoubles = new double[D+1];
				valueDoubles[0] = 1; /* ones padding -- 1 in X_0 for each data point*/
				while(j< (D+1)){
					valueDoubles[j] = Double.parseDouble(valueStrings[j-1]);
					j++;
					z++;
				}
				Point pt = new Point(valueDoubles);
				data[i] = pt;
				j=1;
				i++;
			}
			if ((z != (N*D)) || (strLine != null)){
				System.err.println("Err: Number of values in file differs from NxD in file "+dataFile);
				return null;
			}
		}catch(Exception e){
			//System.err.println("2");
			System.err.println("Error: " + e.getMessage());
			//System.err.println("2!");
			return null;
		}
		return data;
	}
	
	//reads target files
	public static void setTargets(Point[] data, int N, int D, String targetFile){
		try{
			@SuppressWarnings("resource")
			FileInputStream fstream = new FileInputStream(targetFile);
			DataInputStream in = new DataInputStream(fstream);
			@SuppressWarnings("resource")
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			
			int i=0,j=0,z=0;
			while(((strLine = br.readLine()) != null) && i< N){
				String[] valueStrings = strLine.split(" ");
				if(valueStrings.length != C){
					System.err.println("Not enough doubles on line " + i + " of the target set in " + targetFile);
					return;
				}
				double[] target = new double[C];
				while(j< C){
					target[j] = Double.parseDouble(valueStrings[j]);
					j++;
					z++;
				}
				data[i].setTarget(target);
				j=0;
				i++;
			}
			if ((z != (N*C)) || (strLine != null)){
				System.err.println("Err: Number of values in file differs from NxC in file "+targetFile);
				return;
			}
		}catch(Exception e){
			//System.err.println("3");
			System.err.println("Error: " + e.getMessage());
			return;
		}
	}
	/* innerProduct
	 * Calculates the inner product of two vectors a and b
	 */
	private static double innerProduct(double[] a,double[] b){
		int length = a.length;
		double sum = 0;
		for (int i=0; i<length; i++){
			sum += a[i] * b[i];
		}
		return sum;
	}
	private static double[][] outerProduct(double[] a, double[] b){
		int alen=a.length;
		int blen=b.length;
		double[][] result = new double[alen][blen];
		
		for (int i=0;i<alen;i++){
			for (int j=0;j<blen;j++){
				result[i][j]= a[i]*b[j];
			}
		}
		return result;
	}
	private static double[][] matrixMultiply(double[][] a, double[][] b){
		int aRows = a.length;
		int aCols = a[0].length;
		int bRows = b.length;
		int bCols = b[0].length;
		
		if(aCols != bRows){
			System.err.println("number of columns and rows aren't valid");
			System.exit(-1);
		}
		double[][] result = new double[aRows][bCols];
		for (int i=0; i<aRows; i++){
			for (int j=0; j<bCols; j++){
				for (int k=0; k<aCols; k++){
					result[i][j] = a[i][k] * b[k][j];
				}
			}
		}
		return result;
	}
	private static double[][] matrixAdd(double[][] a, double[][] b){
		int aRows = a.length;
		int aCols = a[0].length;
		int bRows = b.length;
		int bCols = b[0].length;
		
		if (aRows != bRows || aCols!=bCols){
			System.err.println("Cannot add matrices with non-equal dimensions");
			System.exit(-1);
		}
	
		double[][] result = new double[aRows][aCols];
		
		for (int i=0;i<aRows;i++){
			for (int j=0; j<aCols; j++){
				result[i][j] = a[i][j] + b[i][j];
			}
		}
		return result;
	}
	private static double[] matrixByVector(double[][] a, double[] b){
		int aRows = a.length;
		int aCols = a[0].length;
		int bLeng = b.length;
		
		if(aCols != bLeng){
			System.err.println("number of columns and rows aren't valid");
			System.exit(-1);
		}
		
		double[] result = new double[aRows];
		
		for (int i=0; i<aRows; i++){
			for (int k=0; k<aCols; k++){
				result[i] += a[i][k] * b[k];
			}
		}
		return result;
	}
}
