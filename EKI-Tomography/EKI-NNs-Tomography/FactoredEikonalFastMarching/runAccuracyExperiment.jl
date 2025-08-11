# function runExperimentAndWriteResults(kappaSquared::Array{Float64},h::Array{Float64,1},src::Array{Int64,1},n::Array{Int64,1},T_exact::Array{Float64},WU::Float64 )
function runExperimentAndWriteResults(kappaSquared::Array{Float64}, h::Array{Float64,1}, src::Array{Int64,1},n::Array{Int64,1}, flag::Int64)
	Omega = zeros(2*length(n));
	Omega[2:2:end] = (n.-1).*h;
	Mesh = getRegularMesh(Omega,n.-1);
	##########################################################

	kappaSquared = 1 ./ kappaSquared.^2
	
	pMem = getEikonalTempMemory(n);	# 1.to be known? 2.all index 3.all value
	pEik = getEikonalParam(Mesh,kappaSquared,src,true); # 1.Mesh 2.κ 3.source location 4.high-order? 5.τ1 6.ordering 7.operator
	pEik.T1 = zeros(Float64,tuple(n...));
	# tictimes = 1000000;
	# times = div(tictimes, prod(n)) .+ 1;	# times = 20
	# println("Timing is based on averaged ",times," runs.");
	# tt = time_ns()
	# for k=1:times
	solveFastMarchingUpwindGrad(pEik,pMem);

	# flag = 2
	# if flag == 1
	# 	solveFastMarchingUpwindGrad(pEik,pMem);
	# 	# end
	# 	# t1 = (time_ns() - tt)/(10^9)
	# 	# t1 = t1/times;

	# 	# τ1 = copy(pEik.T1)
	# 	# return τ1
	# else 
	# 	# tt = time_ns()
	# 	# for k=1:times	
	# 	pEik.HO = true;
	# 	solveFastMarchingUpwindGrad(pEik,pMem);
	# 	# end
	# 	# t2 = (time_ns() - tt)/(10^9)
	# 	# t2 = t2/times;

	# 	# τ2 = copy(pEik.T1)
	# 	# return τ2
	# end

	if Mesh.dim==2
		selfMultiplyWithAnalyticSolution2D(Mesh.n.+1,Mesh.h,src,pEik.T1);
	else
		selfMultiplyWithAnalyticSolution3D(Mesh.n.+1,Mesh.h,src,pEik.T1)
	end

	T1 = copy(pEik.T1)
	return T1
end