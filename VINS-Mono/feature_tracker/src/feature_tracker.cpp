#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

//简单给出是否是边界上的点
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

//根据跟踪点的状态转化矩阵
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}
// 先对跟踪点forw_pts按跟踪次数降排序, 然后依次选点, 
// 选一个点, 在mask中将该点周围一定半径的区域设为0, 后面不再选取该区域内的点
// . 有点类似与non-max suppression, 但区别是这里保留track_cnt最高的点.
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    //要求次数大于255应该是
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}
//处理进来的图像，定义时间
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    
	  //读取完之后如果图像太亮或太黑（EQUALIZE=1），使用createCLAHE对图像进行自适应直方图均衡化，
	  //如果图像正常，设置成当前图像。在读取图像的时候进行光流跟踪和特征点的提取
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;
// 这里涉及到3个img(prev_img, cur_img, forw_img)和
// pts(prev_pts,cur_pts, forw_pts)，两者是相似的。
// 刚开始看不是太好理解，cur和forw分别是LK光流跟踪的前后两帧
// ，forw才是真正的“当前”帧，cur实际上是上一帧，而prev是上一次发布的帧，
// 它实际上是光流跟踪以后，prev和forw根据Fundamental Matrix做RANSAC剔除outlier用的，
// 也就是rejectWithF()函数

    if (forw_img.empty())//当前进来是空的意味着前面的都没有，全部初始化
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }
    //处理当前的，清空当前的
    forw_pts.clear();
    //如果上一帧有点
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
// 	调用calcOpticalFlowPyrLK()跟踪cur_pts到forw_pts,根据status,把跟踪失败的点剔除
//文档上是这个函数：使用迭代的Lucas-Kanade结合金字塔点做稀疏特征的光流匹配
// Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
// output status vector (of unsigned chars); 
// each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
//status 会输出对应特征是否在光流中正确对应，对应是1，否则是0
//最后一个参数是minEigThreshold，用来提出点时的一个阈值
	
	cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
//判断光流法跟踪的当前点，那些跟踪上的是否是在边界上
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
//根据跟踪上的点调整整体的结构，将下面这些结构全部重新调整
//只保留后面跟上的点，前面帧中跟不上的就动态删除掉

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;
//判断是否发布特征点
    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
// 	在mask中不为0的区域,调用goodFeaturesToTrack提取新的角点n_pts,
// 	通过addPoints()函数push到forw_pts中, id初始化-1,track_cnt初始化为1.
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
	    //完成脚点提取
	    //Determines strong corners on an image.

//	    The function finds the most prominent corners in the image or 
// 	    in the specified image region, as described in [170]
	    // Function calculates the corner quality measure at every source image pixel using the cornerMinEigenVal or cornerHarris .
// Function performs a non-maximum suppression (the local maximums in 3 x 3 neighborhood are retained).
// The corners with the minimal eigenvalue less than 𝚚𝚞𝚊𝚕𝚒𝚝𝚢𝙻𝚎𝚟𝚎𝚕⋅maxx,yqualityMeasureMap(x,y) are rejected.
// The remaining corners are sorted by the quality measure in the descending order.
// Function throws away each corner for which there is a stronger corner at a distance less than maxDistance.
// mask	Optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), 
// it specifies the region in which the corners are detected.
//以上是文档原文，简单来说这个用来寻找特定区域最稳定的焦点，mask可以用来标示这个特定区域
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
// 增加特征点	n_pts为0 的时候其实没得选只能加，要求满足的个数打不到，就只能加，达到个数就可以筛选
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
   
 //cur->prev forw->cur
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}
// prev_pts和forw_pts做ransac剔除outlier.(实际就是调用了findFundamentalMat函数)
// , 在光流追踪成功就记被追踪+1，数值代表被追踪的次数，数值越大，说明被追踪的就越久
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
//Calculates a fundamental matrix from the corresponding points in two images.
//根据两队点算F,以及status再次筛除一些点
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}
//对特征点去畸变，
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //据说下面这个畸变方法不靠谱，不过我还没有尝试
    //去畸变还有remap等函数
    //https://blog.csdn.net/qq_25254777/article/details/79282403
    //https://blog.csdn.net/qq_20289205/article/details/78632542
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    //计算速度
    //非空的话就找到对应的特征点求速度
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())//为啥最后一个是0？
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
