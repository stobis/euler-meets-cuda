#ifndef EMC_COMMONS_COMMONS_H_
#define EMC_COMMONS_COMMONS_H_

class Logger
{
public:
  virtual void log(std::string info) = 0;
  virtual void log(std::string info, long double time) = 0;
};

class VoidLogger : public Logger
{
public:
  virtual void log(std::string info) {}
  virtual void log(std::string info, long double time = 0) {}
};
static VoidLogger void_logger;

#endif // EMC_COMMONS_COMMONS_H_
