#define EPSILON 300
//#define MANGLING

#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <cmath>
#include "./Eigen/Dense"
#include "./Eigen/SVD"
#include <./GL/freeglut.h>

using namespace std;
using namespace Eigen;

bool compX(const Vector3d& v1, const Vector3d& v2) {
	if (v1[0] == v2[0]) {
		if (v1[1] == v2[1]) {
			return v1[2] < v2[2];
		}
		return v1[1] < v2[1];
	}
	return v1[0] < v2[0];
}

bool compY(const Vector3d& v1, const Vector3d& v2) {
	if (v1[1] == v2[1]) {
		if (v1[0] == v2[0]) {
			return v1[2] < v2[2];
		}
		return v1[0] < v2[0];
	}
	return v1[1] < v2[1];
}

bool compZ(const Vector3d& v1, const Vector3d& v2) {
	if (v1[2] == v2[2]) {
		if (v1[0] == v2[0]) {
			return v1[1] < v2[1];
		}
		return v1[0] < v2[0];
	}
	return v1[2] < v2[2];
}

class PointCloud {
private:
	vector<Vector3d> cloud;
	int volume;

public:
	PointCloud(const int vol = 0) {
		this->volume = vol;
	}

	void instantiate(const std::vector<Vector3d>& points) {
		cloud = vector<Vector3d>(points);
	}

	vector<Vector3d> getCloudCopy() {
		return this->cloud;
	}

	const int& getVolume() const {
		return this->volume;
	}

#ifdef MANGLING
	void writeToFile() {
		ofstream output;
		output.open("mangled.txt");
		MatrixXd trans1(3, 4);
		MatrixXd trans2(3, 4);
		MatrixXd trans3(3, 4);
		trans1 << 0.707, -0.707, 0, 10, 0.707, 0.707, 0, -0.909, 0, 0, 1, 2.35;
		trans2 << 0.939, 0.0, 0.342, -5.0, 0.0, 1.0, 0.0, 0.75, -0.342, 0.0, 0.939, 17.0;
		trans3 << 1.0, 0.0, 0.0, 0.0, 0.0, sqrt(3.0)/2.0, 1.0/2.0, 280.0, 0.0, -1.0/2.0, sqrt(3.0)/2.0, 0.0;
		this->applyTransformation(trans3);
		output << this->volume << "\n";
		for (int i = 0; i < this->volume; i++) {
			output << this->cloud.at(i)[0] << " " << this->cloud.at(i)[1] << " " << this->cloud.at(i)[2] << "\n";
		}
		output.close();
	}
#endif

	void applyTransformation(const MatrixXd& transformation) {
		for (Vector3d& vec : cloud) {
			//Matrix3d R = 
			vec = transformation.block<3, 3>(0, 0) * vec;
			vec += transformation.block<3, 1>(0, 3);
		}
	}
};

struct Node {
	Vector3d point;
	int pointId;
	Node* left;
	Node* right;

	Node(Vector3d p, int id, Node* left, Node* right) {
		this->point = p;
		this->pointId = id;
		this->left = left;
		this->right = right;
	}

	~Node() {
		if (this->left != nullptr) delete this->left;
		if (this->right != nullptr) delete this->right;
	}
};

class KdTree {
private:
	PointCloud* prototype;
	Node* root;
	int dim;

	int medianFit(const vector<Vector3d>& vec, int compDim) {
		int index = 0;
		int counter = 1;
		for (int i = 1; i < vec.size(); i++) {
			if (vec.at(i)[compDim] > vec.at(i - 1)[compDim]) {
				counter++;
			}
		}
		if (counter == 1) {
			return index;
		}
		else {
			index = vec.size() / 2;
			while (index > 0 && vec.at(index - 1)[compDim] == vec.at(index)[compDim]) {
				index--;
			}
			return index;
		}
	}

	Node* build(vector<Vector3d> set, int depth) {
		if (set.empty()) {
			return nullptr;
		}
		else if (set.size() == 1) {
			return new Node(set.back(), 0, nullptr, nullptr);
		}
		else {
			switch (depth % this->dim)
			{
			case 0: {
				sort(set.begin(), set.end(), compX);
				break;
			}
			case 1: {
				sort(set.begin(), set.end(), compY);
				break;
			}
			case 2: {
				sort(set.begin(), set.end(), compZ);
				break;
			}
			default:
				break;
			}
			int medianIndex = medianFit(set, depth % this->dim);
			vector<Vector3d> newV1(medianIndex);
			copy(set.begin(), set.begin() + medianIndex, newV1.begin());
			vector<Vector3d> newV2(set.size() - medianIndex - 1);
			vector<Vector3d>::iterator it = set.begin() + medianIndex + 1;
			if (it >= set.end()) {
				return new Node(set.at(medianIndex), medianIndex, build(newV1, depth + 1), nullptr);
			}
			copy(it, set.end(), newV2.begin());
			return new Node(set.at(medianIndex), medianIndex, build(newV1, depth + 1), build(newV2, depth + 1));
		}
	}

	Vector3d* search(Node* current, const Vector3d& point, double& dist, int depth) {
		if (current == nullptr) {
			return nullptr;
		}
		
		int curDim = depth % this->dim;
		Vector3d* closestPoint = nullptr;

		if (point[curDim] < current->point[curDim]) {
			closestPoint = search(current->left, point, dist, depth + 1);
			if (dist >= pow(point[curDim] - current->point[curDim], 2)) {
				double curDist = dist;
				Vector3d* alternativePoint = search(current->right, point, dist, depth + 1);
				if (dist < curDist) {
					closestPoint = alternativePoint;
				}
			}
		}
		else {
			closestPoint = search(current->right, point, dist, depth + 1);
			if (dist >= pow(point[curDim] - current->point[curDim], 2)) {
				double curDist = dist;
				Vector3d* alternativePoint = search(current->left, point, dist, depth + 1);
				if (dist < curDist) {
					closestPoint = alternativePoint;
				}
			}
		}

		double thisDist = pow(point[0] - current->point[0], 2) + pow(point[1] - current->point[1], 2) + pow(point[2] - current->point[2], 2);
		if (thisDist < dist) {
			dist = thisDist;
			return &current->point;
		}
		else {
			return closestPoint;
		}
	}

public:
	KdTree(PointCloud* proto, int d) : root(nullptr), prototype(proto), dim(d) {
		this->root = build(this->prototype->getCloudCopy(), 0);
	}

	~KdTree() { delete this->root; }

	void searchNearest(const Vector3d& point, Vector3d& targPoint, double& distance) {
		double dist = numeric_limits<double>::infinity();
		targPoint = *search(this->root, point, dist, 0);
		distance = dist;
	}
};

class ICP {
private:
	PointCloud* source;
	PointCloud* target;
	KdTree* kdtree;

	MatrixXd findTransformation(const MatrixXd& target, const MatrixXd& source) {
		Vector3d sourceCentroid(0, 0, 0);
		Vector3d targetCentroid(0, 0, 0);

		MatrixXd centeredSource(source.rows(), 3);
		MatrixXd centeredTarget(target.rows(), 3);

		MatrixXd H;
		MatrixXd U;
		MatrixXd V;
		Matrix3d R;
		Vector3d t;

		MatrixXd transformation(3, 4);

		for (int i = 0; i < target.rows(); i++) {
			sourceCentroid += source.block<1, 3>(i, 0).transpose();
			targetCentroid += target.block<1, 3>(i, 0).transpose();
		}
		sourceCentroid /= source.rows();
		targetCentroid /= target.rows();
		for (int i = 0; i < target.rows(); i++) {
			centeredSource.block<1, 3>(i, 0) = source.block<1, 3>(i, 0) - sourceCentroid.transpose();
			centeredTarget.block<1, 3>(i, 0) = target.block<1, 3>(i, 0) - targetCentroid.transpose();
		}

		H = centeredSource.transpose() * centeredTarget;

		JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
		U = svd.matrixU();
		V = svd.matrixV();

		R = V * U.transpose();
		if (R.determinant() < 0) {
			MatrixXd diag(3, 3);
			diag << 1, 0, 0, 0, 1, 0, 0, 0, -1;
			R = V * diag * U.transpose();
		}

		t = targetCentroid - R * sourceCentroid;

		transformation.block<3, 3>(0, 0) = R;
		transformation.block<3, 1>(0, 3) = t;

		return transformation;
	}

public:
	ICP(PointCloud* t, PointCloud* s = nullptr): target(t), source(s) {
		try {
			this->kdtree = new KdTree(target, 3);
		}
		catch (exception ex) {
			ex.what();
		}
	}

	~ICP() {
		delete this->kdtree;
	}

	void icp() {
		double error = numeric_limits<double>::infinity();
		MatrixXd corrMatrix(3, this->target->getVolume());
		MatrixXd sourceMatrix(3, this->source->getVolume());
		vector<Vector3d> sourceCloud;
		while(error>= EPSILON) {
			sourceCloud = this->source->getCloudCopy();
			double curError = 0;

			for (int j = 0; j < this->target->getVolume(); j++) {
				Vector3d sPoint = sourceCloud.at(j);
				Vector3d corrPoint;
				double distance;
				kdtree->searchNearest(sPoint, corrPoint, distance);
				sourceMatrix.block<3, 1>(0, j) = sPoint;
				corrMatrix.block<3, 1>(0, j) = corrPoint;
				curError += distance;
			}

			if (curError < error && error - curError >= EPSILON) {
				error = curError;
				MatrixXd transformation = findTransformation(corrMatrix.transpose(), sourceMatrix.transpose());
				this->source->applyTransformation(transformation);
			}
			else {
				break;
			}
		}
		cout << "Algorithm has been terminated\n";
		delete this;
	}
};

PointCloud ps1;
PointCloud ps2;
double position[6] = { 0.0, -130.0, 100.0, 0.0, -129.0, 99.35};

void render(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(position[0], position[2], position[1],
		position[3], position[5], position[4],
		0.0, 1.0, 0.0);

	glColor3d(0.0, 1.0, 0.0);
	for (const Vector3d& vec : ps1.getCloudCopy()) {
		glPushMatrix();
		glTranslated(vec[0], vec[2], vec[1]);
		glutSolidCube(0.45);
		glRotated(45.0, 0.0, 1.0, 0.0);
		glRotated(45.0, 1.0, 0.0, 0.0);
		glPopMatrix();
	}

	glColor3d(1.0, 0.0, 0.0);
	for (const Vector3d& vec : ps2.getCloudCopy()) {
		glPushMatrix();
		glTranslated(vec[0], vec[2], vec[1]);
		glutSolidSphere(0.3, 20, 20);
		glPopMatrix();
	}

	glFlush();
	glutSwapBuffers();
}

void resize(int w, int h) {
	if (h == 0)
		h = 1;
	float ratio = w * 1.0 / h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, w, h);
	gluPerspective(45.0f, ratio, 0.1f, 500.0f);
	glMatrixMode(GL_MODELVIEW);
}

void movement(unsigned char key, int x, int y) {
	double dx = position[0] - position[3];
	double dy = position[2] - position[5];
	double dz = position[1] - position[4];
	double length = sqrt(dx * dx + dy * dy + dz * dz);
	dx /= length;
	dy /= length;
	dz /= length;

	switch (key) {
	case 'w':
	{
		position[0] -= dx * 3.0;
		position[1] -= dz * 3.0;
		position[2] -= dy * 3.0;
		position[3] -= dx * 3.0;
		position[4] -= dz * 3.0;
		position[5] -= dy * 3.0;
		break;
	}
	case 's': {
		position[0] += dx * 3.0;
		position[1] += dz * 3.0;
		position[2] += dy * 3.0;
		position[3] += dx * 3.0;
		position[4] += dz * 3.0;
		position[5] += dy * 3.0;
		break;
	}
	case 'a':
	{
		position[0] -= dz * 5.0;
		position[3] -= dz * 5.0;
		position[1] += dx * 5.0;
		position[4] += dx * 5.0;
		break;
	}
	case 'd':
	{
		position[0] += dz * 5.0;
		position[3] += dz * 5.0;
		position[1] -= dx * 5.0;
		position[4] -= dx * 5.0;
		break;
	}
	case 'e':
	{
		position[2] += 3.0;
		position[5] += 3.0;
		break;
	}
	case 'q':
	{
		position[2] -= 3.0;
		position[5] -= 3.0;
	}
	}
}

void observation(int key, int x, int y) {
	double dx = position[3] - position[0];
	double dy = position[5] - position[2];
	double dz = position[4] - position[1];

	Vector3d axis(dx, dz, dy);
	axis = axis.cross(Vector3d(0.0, 0.0, 1.0));
	axis.normalize();
	double angle = asin(1) / 36;
	Quaternion<double> quat;

	switch (key) {
	case GLUT_KEY_UP:
	{
		quat = AngleAxis<double>(angle, axis);
		Vector3d rotated = quat.toRotationMatrix() * Vector3d(dx, dz, dy);
		position[3] += rotated[0] - dx;
		position[5] += rotated[2] - dy;
		position[4] += rotated[1] - dz;
		break;
	}
	case GLUT_KEY_DOWN:
	{
		quat = AngleAxis<double>(-angle, axis);
		Vector3d rotated = quat.toRotationMatrix() * Vector3d(dx, dz, dy);
		position[3] += rotated[0] - dx;
		position[5] += rotated[2] - dy;
		position[4] += rotated[1] - dz;
		break;
	}
	case GLUT_KEY_LEFT:
	{
		double newDx = cos(-angle) * dx - sin(-angle) * dz;
		double newDz = sin(-angle) * dx + cos(-angle) * dz;
		position[3] += newDx - dx;
		position[4] += newDz - dz;
		break;
	}
	case GLUT_KEY_RIGHT:
	{
		double newDx = cos(angle) * dx - sin(angle) * dz;
		double newDz = sin(angle) * dx + cos(angle) * dz;
		position[3] += newDx - dx;
		position[4] += newDz - dz;
	}
	}
}

int main(int argc, char** argv) {
	double x1 = 0, y1 = 0, z1 = 0,
		x2 = 0, y2 = 0, z2 = 0;
	int vol = 0;
	string filename1 = "", filename2 = "";

	ifstream input1, input2;
	switch (argc) {
	case 1:
	{
		cout << "Type \"exit\" to stop the program\n\n";
		cout << "Enter the path to a template file: ";
		cin >> filename1;
		if (filename1 == "exit" || cin.eof()) {
			return 0;
		}
	}
	case 2:
	{
		if (filename1 == "") {
			filename1 = argv[1];
			cout << "Type \"exit\" to stop the program\n\n";
		}
		cout << "Enter the path to a file to analyze: ";
		cin >> filename2;
		if (filename2 == "exit" || cin.eof()) {
			return 0;
		}
		break;
	}
	case 3:
	{
		filename1 = argv[1];
		filename2 = argv[2];
	}
	}

	input1.open(filename1);
	input2.open(filename2);
	if (!(input1.is_open() && input2.is_open())) {
		cout << "Error: cannot open file(s).\n";
		system("pause");
		return 0;
	}
	
	input1 >> vol;
	ps1 = PointCloud(vol);
	input2 >> vol;
	if (vol != ps1.getVolume()) {
		input1.close();
		input2.close();
		cout << "Error: point amounts are different.\n";
		system("pause");
		return 0;
	}
	ps2 = PointCloud(vol);

	vector<Vector3d> pVec1;
	vector<Vector3d> pVec2;

	for (int i = 0; i < vol; i++) {
		input1 >> x1 >> y1 >> z1;
		input2 >> x2 >> y2 >> z2;
		pVec1.push_back(Vector3d(x1, y1, z1));
		pVec2.push_back(Vector3d(x2, y2, z2));
	}
	input1.close();
	input2.close();

	ps1.instantiate(pVec1);
	ps2.instantiate(pVec2);

#ifdef MANGLING
	ps1.writeToFile();
#else

	ICP* solver = new ICP(&ps1, &ps2);

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(560, 240);
	glutInitWindowSize(800, 600);
	glutCreateWindow("ICP algorithm");

	glClearColor(0.03, 0.19, 0.23, 0.3);
	glutDisplayFunc(render);
	glutReshapeFunc(resize);
	glutIdleFunc(render);
	glutKeyboardFunc(movement);
	glutSpecialFunc(observation);
	glEnable(GL_DEPTH_TEST);

	thread t(&ICP::icp, solver);
	t.detach();
	glutMainLoop();
#endif
}
